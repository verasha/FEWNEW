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

from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux

from few.utils.geodesic import get_fundamental_frequencies

from few.utils.constants import YRSID_SI

import os
import sys

# Changing directory for VSCode notebook
os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/')

import localgit.FEWNEW.work.GWfuncs_backup2 as GWfuncs_backup2
# import loglike
# import modeselector
# import parismc
import dynesty
# import gc
import pickle
import cupy as cp

# tune few configuration
cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

# GPU configuration 
use_gpu = True
force_backend = "cuda12x"  
dt = 10     # Time step
T = 0.25     # Total time

print('Initializing waveform generator...')
# keyword arguments for inspiral generator 
inspiral_kwargs={
        "func": 'KerrEccEqFlux',
        "DENSE_STEPPING": 0, #change to 1/True for uniform sampling
        "include_minus_m": False, 
}

# keyword arguments for inspiral generator 
amplitude_kwargs = {
    "force_backend": "cuda12x" # Force GPU
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
}

print("Creating GenerateEMRIWaveform class...")
# Kerr eccentric flux
waveform_gen = GenerateEMRIWaveform(
    FastKerrEccentricEquatorialFlux, 
    frame='detector',
    inspiral_kwargs=inspiral_kwargs, 
    amplitude_kwargs=amplitude_kwargs, 
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=use_gpu
)

print('Done initializing waveform generator.')

print("Creating GravWaveAnalysis class...")
gwf = GWfuncs_backup2.GravWaveAnalysis(T, dt)

# Source parameters
m1 = 1e6
m2 = 3e1
a = 0.7
p0 = 7.5
e0 = 0.4 
xI0 = 1.0
dist = 0.5 # Gpc
# Polar and azimuthal angles .. detector frame
# S = Solar system barycenter
# K = spin angular momentum of the MBH
qS = 0.5 
phiS = 1 
qK = 1 #fixed
phiK = phiS + np.pi/3
# Phases
Phi_phi0 = 0.4
Phi_theta0 = 0.0 # equatorial
Phi_r0 = 0.5

print("Generate data...")
data = waveform_gen(m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, T=T, dt=dt)

print("Defining functions for loglike and prior")
def loglike(params):
    logm1, logm2, a, p0, e0 = params
    m1 = 10**logm1
    m2 = 10**logm2

    htemp = waveform_gen(m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
                            Phi_phi0, Phi_theta0, Phi_r0, T=T, dt=dt)
            
    res = data - htemp
    res_f = gwf.freq_wave(res)
    inner_res = gwf.inner(res_f, res_f)
    log_like = -0.5 * inner_res.get()
    return log_like

def prior_transform(utheta):
    ulogm1, ulogm2, ua, up0, ue0 = utheta

    logm1_range = (5.999851037559485, 6.000148962440515)
    logm2_range = (1.4770378504576676, 1.4772046589816572)
    a_range =  (0.6997315972442014, 0.7002684027557985)
    p0_range =  (7.498519407697586, 7.501480592302414)
    e0_range =  (0.3999359060325742, 0.40006409396742587)


    # All uniform
    logm1 = (logm1_range[1] - logm1_range[0]) * ulogm1 + logm1_range[0]
    logm2 = (logm2_range[1] - logm2_range[0]) * ulogm2 + logm2_range[0]
    a = (a_range[1] - a_range[0]) * ua + a_range[0]
    p0 = (p0_range[1] - p0_range[0]) * up0 + p0_range[0]
    e0 = (e0_range[1] - e0_range[0]) * ue0 + e0_range[0]

    return logm1, logm2, a, p0, e0


print('Done setting up loglike and prior_transform functions.')
print('Starting nested sampling...')
rstate = np.random.default_rng(7)
with dynesty.pool.Pool(16, loglike, prior_transform) as pool:
    dsampler = dynesty.DynamicNestedSampler(
        pool.loglike,  
        pool.prior_transform,
        ndim=5,
        bound='multi',
        sample='rwalk',
        rstate=rstate,
        nlive=500
    )
    dsampler.run_nested(checkpoint_file='ns_intrinsic2.save')

print('Done nested sampling.')
print('Saving results to pickle...')
results = dsampler.results
with open('ns_intrinsic2.pkl', 'wb') as f:
    pickle.dump(results, f)