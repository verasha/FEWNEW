print("Starting imports...")
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, float64, complex128
from numba.cuda import jit as cuda_jit
import math

print("Importing few...")
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

print("Importing GWfuncs...")
import GWfuncs
# import gc
# import pickle
print("Importing cupy...")
import cupy as cp

print("Configuring few...")
# tune few configuration
cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

print("Importing dynesty...")
import dynesty

for backend in ["cpu", "cuda11x", "cuda12x", "cuda", "gpu"]: 
    print(f" - Backend '{backend}': {"available" if few.has_backend(backend) else "unavailable"}")  

# GPU configuration and missing variables
use_gpu = True
dt = 10     # Time step
T = 1.0     # Total time

# Check GPU availability
if not cuda.is_available():
    print("Warning: CUDA not available, falling back to CPU")
    use_gpu = False

print("Setting up waveform generator...")
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
    "force_backend": "cuda12x" # Force GPU
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
    "pad_output": True,
    # "use_gpu" : use_gpu
}

print("Creating FastKerrEccentricEquatorialFlux...")
# Kerr eccentric flux
waveform_gen = FastKerrEccentricEquatorialFlux(
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=use_gpu,
)

def loglike(params):
    """Log-likelihood function for the nested sampling."""
    # Convert theta to parameters
    m1, m2, a, p0, e0, xI0, theta, phi, dist = params
    # print(f"=== Parameters: m1={m1:.2e}, m2={m2:.2e}, a={a:.3f}, p0={p0:.2f}, e0={e0:.3f}, theta={theta:.3f}, phi={phi:.3f}, dist = {dist:.3f} ===")

    # Generate template waveform with the current parameters
    h_temp = waveform_gen(m1, m2, a, p0, e0, xI0, theta, phi, dist=dist, dt=dt, T=T)
    # print(f"h_temp shape: {h_temp.shape}")
    
    # Calculate the factor for normalization
    factor = gwf.dist_factor(dist, m2)

    # Generate trajectory
    (t, p, e, x, Phi_phi, Phi_theta, Phi_r) = traj(m1, m2, a, p0, e0, xI0, T=T, dt=dt)
    # print(f"Trajectory length: {len(t)}")
    t_gpu = cp.asarray(t)

    # Get amplitudes along trajectory
    teuk_modes = amp(a, p, e, x)
    # print(f"teuk_modes shape: {teuk_modes.shape}")

    # Get Ylms
    ylms = ylm_gen(amp.unique_l, amp.unique_m, theta, phi).copy()[amp.inverse_lm]

    # Calculate power for all modes
    m0mask = amp.m_arr_no_mask != 0
    total_power = gwf.calc_power(teuk_modes, ylms, m0mask)

    # Get mode labels
    mode_labels = [f"({l},{m},{n})" for l,m,n in zip(amp.l_arr, amp.m_arr, amp.n_arr)]

    # Change num of selected modes here 
    M_mode = 10

    # Get top M indices
    top_indices_gpu = gwf.xp.argsort(total_power)[-M_mode:][::-1]  # Top M indices in descending order
    top_indices = top_indices_gpu.get().tolist()  # Convert to CPU list only once

    # Pick modes based on top M power contributions
    mp_modes = [mode_labels[idx] for idx in top_indices]
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
    # print(f"rho_m: {rho_m}")
    
    # Calculate Xm 
    X_modes = gwf.Xmstat(h_true, waveform_per_mode, rho_m)
    # print(f"X_modes: {X_modes}")

    # Calculate X_scalar
    Xdotrho = gwf.xp.sum(X_modes * rho_m)
    rho_norm = gwf.xp.sqrt(gwf.xp.sum(rho_m**2))
    X_scalar = Xdotrho / rho_norm
    # print(f"X_scalar: {X_scalar:.4f}")

    X_check = gwf.Xstat(h_true, h_temp)
    # print(f"X_check: {X_check:.4f}")

    # Calculate optimal SNR of most dominant mode by power
    rho_dom_M = gwf.rhostat(waveform_per_mode[0])

    # Calculate total rho 
    rho_tot = gwf.rhostat(h_temp)

    # Calculate alpha with numerical stability
    alpha = rho_dom_M / rho_tot  
    # print(f"alpha: {alpha:.4f}")

    # Calculate beta with numerical checks
    beta_num = 2 * gwf.xp.log(alpha * rho_tot)
    beta_denom = (1-alpha**2) * rho_tot**2 

    beta = beta_num / beta_denom
    # print(f"beta: {beta:.4f}")

    # Calculate chi sq
    chi_sq = gwf.chi_sq(X_modes, rho_m)
    # print(f"chi_sq: {chi_sq:.4f}")

    # Calculate f statistic 
    f_exp = -0.5 * beta * chi_sq 

    # Overflow protection for large exponentials 
    # TODO: Check for alternate ways that are more accurate? mpmath? logsumexp trick?
    if f_exp > 700:  
        f_exp = 700 # Cap 
    elif f_exp < -700:
        return -np.inf  
    
    f_stat = X_scalar * gwf.xp.exp(f_exp)

    logl_res = float(gwf.xp.real(f_stat).get())
    
    # Check for NaNs 
    if np.isnan(logl_res):
        return -np.inf

    return logl_res 


#Generating data (true)

m1_o = 1e6
m2_o = 1e1
a_o = 0.3
p0_o = 12
e0_o = 0.1
xI_o = 1.0
theta_o = np.pi/3  # polar viewing angle
phi_o = np.pi/4  # azimuthal viewing angle
dist = 1 # Gpc

print("Generating true waveform...")
h_true = waveform_gen(m1_o, m2_o, a_o, p0_o, e0_o, xI_o, theta_o, phi_o, dist=dist, dt=dt, T=1)

print("Creating GWfuncs analysis object...")
N_true = int(len(h_true))  # Use data length as reference
gwf = GWfuncs.GravWaveAnalysis(N=N_true, dt=dt)

print("Initializing trajectory and amplitude generators...")
# Initialize trajectory and amplitude generators
traj = EMRIInspiral(func=KerrEccEqFlux, force_backend="cuda12x", use_gpu=use_gpu)
amp = AmpInterpKerrEccEq(force_backend="cuda12x")
interpolate_mode_sum = InterpolatedModeSum(force_backend="cuda12x", pad_output= True)
ylm_gen = GetYlms(include_minus_m=False, force_backend="cuda12x")

## PRIOR: masses log-uniform
def prior_transform(utheta):
      um1, um2, ua, up0, ue0, uxI0, utheta_angle, uphi, udist = utheta

      # Parameter limits
      m1lim = [9.99e5, 1.001e6]
      m2lim = [9.99, 10.01]
      alim = [0.2997, 0.3003]
      p0lim = [11.988, 12.012]
      e0lim = [9.99e-2, 0.1001]
      xI0lim = [1.0, 1.0]
      thetalim = [np.pi / 3 * (0.999), np.pi / 3 * (1.001)]
      philim = [np.pi / 4 * (0.999), np.pi / 4 * (1.001)]
      distlim = [0.999, 1.001]  # Distance in Gpc

      # Log-uniform for masses
      m1 = 10**(np.log10(m1lim[0]) + um1 * (np.log10(m1lim[1]) - np.log10(m1lim[0])))
      m2 = 10**(np.log10(m2lim[0]) + um2 * (np.log10(m2lim[1]) - np.log10(m2lim[0])))

      # Uniform for other parameters
      a = (alim[1] - alim[0]) * ua + alim[0]
      p0 = (p0lim[1] - p0lim[0]) * up0 + p0lim[0]
      e0 = (e0lim[1] - e0lim[0]) * ue0 + e0lim[0]
      xI0 = 1.0  # Fixed value
      theta = (thetalim[1] - thetalim[0]) * utheta_angle + thetalim[0]
      phi = (philim[1] - philim[0]) * uphi + philim[0]
      dist = (distlim[1] - distlim[0]) * udist + distlim[0]

      return m1, m2, a, p0, e0, xI0, theta, phi, dist

rstate = np.random.default_rng(7)
with dynesty.pool.Pool(16, loglike, prior_transform) as pool:
    dsampler = dynesty.NestedSampler(
        loglike,  
        prior_transform,
        ndim=9,
        bound='multi',
        sample='rwalk',
        rstate=rstate
    )
    dsampler.run_nested(checkpoint_file='dynestystatic.save')