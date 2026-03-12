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
# import gc
# import pickle
import cupy as cp

import scipy.optimize as opt
import emcee



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
dist = 2 # Gpc
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
    calc_loglike = -0.5 * inner_res

    # return calc_loglike
    return float(cp.asnumpy(calc_loglike)) 

def log_prior(params):    
    bounds = [
        (5.9999650899e+00, 6.0000349101e+00),  # logm1
        (1.4771002185e+00, 1.4771422909e+00),  # logm2
        (6.9993760164e-01, 7.0006239836e-01),  # a
        (7.4996582212e+00, 7.5003417788e+00),  # p0
        (3.9998409234e-01, 4.0001590766e-01),  # e0
    ]
    
    for param, (lower, upper) in zip(params, bounds):
        if not (lower <= param <= upper):
            return -np.inf
    
    return 0.0

def logprob(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike(params)

print("Setting up emcee sampler...")
# Initialize walkers 
Ndim = 5 
Nwalker = 32

# Find MLE to initialize walkers
nll = lambda *args: -loglike(*args)
result = opt.minimize(nll, [np.log10(m1), np.log10(m2), a, p0, e0])
pos = [result['x']+1.e-4*np.random.randn(Ndim) for i in range(Nwalker)]

# HDF5 file to save intermediate results
backend = emcee.backends.HDFBackend('emcee_backend.h5')
# Reset to clear any previous runs
backend.reset(Nwalker, Ndim)



sampler = emcee.EnsembleSampler(Nwalker, Ndim, logprob, backend=backend)
sampler.run_mcmc(pos, int(5e4), progress=True)

