import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, float64, complex128
from numba.cuda import jit as cuda_jit
import math

import few

from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import KerrEccEqFlux
from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
from few.summation.interpolatedmodesum import InterpolatedModeSum 


from few.utils.ylm import GetYlms

from few import get_file_manager

from few.waveform import FastKerrEccentricEquatorialFlux

from few.utils.geodesic import get_fundamental_frequencies

import os
import sys

# Change to the desired directory
os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/')

# Add it to Python path
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/')

import GWfuncs
# import gc
# import pickle
import cupy as cp

# tune few configuration
cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

import dynesty

# GPU configuration and missing variables
use_gpu = True
dist = 1.0  # Distance in Gpc
dt = 10     # Time step
T = 1.0     # Total time

# Check GPU availability
if not cuda.is_available():
    print("Warning: CUDA not available, falling back to CPU")
    use_gpu = False

# keyword arguments for inspiral generator 
inspiral_kwargs={
        "func": 'KerrEccEqFlux',
        "DENSE_STEPPING": 0, #change to 1/True for uniform sampling
        "include_minus_m": False, 
        "use_gpu" : use_gpu,
        "force_backend": "cuda12x"  # Force GPU
}

# keyword arguments for inspiral generator 
amplitude_kwargs = {
    "force_backend": "cuda12x",  # Force GPU
    # "use_gpu" : use_gpu
}

# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "force_backend": "cuda12x",  # Force GPU
    # "assume_positive_m": True  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs = {
    "force_backend": "cuda12x",  # Force GPU
    "pad_output": False,
    # "use_gpu" : use_gpu
}

# Kerr eccentric flux
waveform_gen = FastKerrEccentricEquatorialFlux(
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=use_gpu,
)

#Generating data (true)

m1_o = 1e5
m2_o = 1e1
a_o = 0.3
p0_o = 40
e0_o = 0.1
xI_o = 1.0
theta_o = np.pi/3  # polar viewing angle
phi_o = np.pi/2  # azimuthal viewing angle

h_true = waveform_gen(m1_o, m2_o, a_o, p0_o, e0_o, xI_o, theta_o, phi_o, dist=dist, dt=dt, T=1)

N = int(len(h_true)) 
gwf = GWfuncs.GravWaveAnalysis(N=N,dt=dt)

def loglike(params):
    """Log-likelihood function for the nested sampling."""
    # Convert theta to parameters
    m1, m2, a, p0, e0, xI0, theta, phi = params

    # Generate template waveform with the current parameters
    h_temp = waveform_gen(m1, m2, a, p0, e0, xI0, theta, phi, dist=dist, dt=dt, T=1)

    # Generate trajectory
    (t, p, e, x, Phi_phi, Phi_theta, Phi_r) = traj(m1, m2, a, p0, e0, xI0, T=T, dt=dt)

    # Get amplitudes along trajectory
    teuk_modes = amp(a, p, e, x)

    # Get Ylms
    ylms = ylm_gen(amp.unique_l, amp.unique_m, theta, phi).copy()[amp.inverse_lm]

    # Calculate power for all modes
    m0mask = amp.m_arr_no_mask != 0
    total_power = gwf.calc_power(teuk_modes, ylms, m0mask)

    # Change num of selected modes here 
    M_mode = 10

    # Get top M indices
    top_indices_gpu = gwf.xp.argsort(total_power)[-M_mode:][::-1]  # Top M indices in descending order
    top_indices = top_indices_gpu.get().tolist()  # Convert to CPU list only once

    # Pick modes based on top M power contributions
    top_indices = [mode_labels.index(mode) for mode in mp_modes]

    # Generate hm_arr for top modes
    waveform_per_mode = []
    for idx in top_indices:
        l = amp.l_arr[idx]
        m = amp.m_arr[idx]
        n = amp.n_arr[idx]

        if m >= 0:
            teuk_modes_single = teuk_modes[:, [idx]]
            ylms_single = ylms[[idx]]
            m_arr = amp.m_arr[[idx]]
        else:
            pos_m_mask = (amp.l_arr == l) & (amp.m_arr == -m) & (amp.n_arr == n)
            pos_m_idx = gwf.xp.where(pos_m_mask)[0][0]
            teuk_modes_single = (-1)**l * gwf.xp.conj(teuk_modes[:, [pos_m_idx]])
            ylms_single = ylms[[idx]]
            m_arr = gwf.xp.abs(amp.m_arr[[idx]]) 

        waveform = interpolate_mode_sum(
            t_gpu, teuk_modes_single, ylms_single,
            traj.integrator_spline_t, traj.integrator_spline_phase_coeff[:, [0, 2]],
            amp.l_arr[[idx]], m_arr, amp.n_arr[[idx]], 
            dt=dt, T=T
        )
        waveform_per_mode.append(waveform / factor)

    # Calculate rho_m
    rho_m = gwf.rhostat_modes(waveform_per_mode)
    # Calculate Xm 
    X_modes = gwf.Xmstat(waveform_per_mode, rho_m)

    # Calculate X_scalar
    Xdotrho = gwf.xp.sum(X_modes * rho_m)
    rho_norm = gwf.xp.sqrt(gwf.xp.sum(rho_m**2))
    X_scalar = Xdotrho / rho_norm

    # Calculate optimal SNR of most dominant mode by power
    rho_dom_M = gwf.rhostat(waveform_per_mode[0])

    # Calculate total rho 
    rho_tot = gwf.rhostat(h_temp)

    # Calculate alpha
    alpha = rho_dom_M / rho_tot

    # Calculate beta 
    beta_num = 2 * gwf.xp.log(alpha * rho_tot)
    beta_denom = (1-alpha**2) * rho_tot**2 
    beta = beta_num / beta_denom

    # Calculate chi sq
    chi_sq = gwf.xp.abs(X_modes - rho_m)**2

    # Calculate f statistic
    f_exp = -0.5 * beta * chi_sq 
    f_stat = X_scalar * gwf.xp.exp(f_exp)

    return f_stat
    

def prior_transform(params):
    ## TODO ##
    return ###


