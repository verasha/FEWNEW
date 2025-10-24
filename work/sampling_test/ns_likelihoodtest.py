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

# Change to the desired directory
os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/')

# Add it to Python path
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/')

import GWfuncs
# import loglike
# import modeselector
# import parismc
# import gc
import pickle
import dynesty
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

# Source parameters
m1 = 1e6
m2 = 3e1
a = 0.7
p0 = 7.5
e0 = 0.4 
xI0 = 1.0 #NOTE: fixed
dist = 0.5 # Gpc
# Polar and azimuthal angles .. detector frame
# S = Solar system barycenter
# K = spin angular momentum of the MBH
qS = 0.5 
phiS = 1 
qK = 1 #NOTE: fixed
phiK = phiS + np.pi/3
# Phases
Phi_phi0 = 0.4
Phi_theta0 = 0.0 # NOTE: fixed
Phi_r0 = 0.5

print('Generating data signal...')
data = waveform_gen(m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, T=T, dt=dt)
print('Done generating data signal.')

print('Setting up GWFuncs...')
gwf = GWfuncs.GravWaveAnalysis(T, dt)
print('Done setting up GWFuncs.')

print('Setting up loglike and prior_transform functions...')
print("We're using wider priors")
def loglike(params):
    logm1, logm2, a, p0, e0, dist, cosqS, phiS, Phi_phi0, Phi_r0 = params
    m1 = 10**logm1
    m2 = 10**logm2
    qS = np.arccos(cosqS)
    phiK = phiS + np.pi/3

    htemp = waveform_gen(m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
                            Phi_phi0, Phi_theta0, Phi_r0, T=T, dt=dt)
            
    res = data - htemp
    res_f = gwf.freq_wave(res)
    inner_res = gwf.inner(res_f, res_f)
    log_like = -0.5 * inner_res.get()
    return log_like

def prior_transform(utheta):
    ulogm1, ulogm2, ua, up0, ue0, udist, ucosqS, uphiS, uPhi_phi0, uPhi_r0 = utheta

    # 3 sigma ranges
    logm1_range = (5.999755966003094, 6.000244033996906)
    logm2_range = (1.4769870513061485, 1.4772554581331763)
    a_range = (0.6995637359288023, 0.7004362640711976)
    p0_range = (7.497572969128289, 7.502427030871711)
    e0_range = (0.39989407441630453, 0.4001059255836955)
    dist_range = (0.4671238045922167, 0.5328761954077833)
    cosqS_range = (0.8175373164938913, 0.9376278072868542)
    phiS_range = (0.8907980491383655, 1.1092019508616344)
    Phi_phi0_range = (0.2971011746225549, 0.5028988253774451)
    Phi_r0_range = (0.4374942669461721, 0.5625057330538279)

    # All uniform
    logm1 = (logm1_range[1] - logm1_range[0]) * ulogm1 + logm1_range[0]
    logm2 = (logm2_range[1] - logm2_range[0]) * ulogm2 + logm2_range[0]
    a = (a_range[1] - a_range[0]) * ua + a_range[0]
    p0 = (p0_range[1] - p0_range[0]) * up0 + p0_range[0]
    e0 = (e0_range[1] - e0_range[0]) * ue0 + e0_range[0]
    dist = (dist_range[1] - dist_range[0]) * udist + dist_range[0]
    cosqS = (cosqS_range[1] - cosqS_range[0]) * ucosqS + cosqS_range[0]
    phiS = (phiS_range[1] - phiS_range[0]) * uphiS + phiS_range[0]
    Phi_phi0 = (Phi_phi0_range[1] - Phi_phi0_range[0]) * uPhi_phi0 + Phi_phi0_range[0]
    Phi_r0 = (Phi_r0_range[1] - Phi_r0_range[0]) * uPhi_r0 + Phi_r0_range[0]

    return logm1, logm2, a, p0, e0, dist, cosqS, phiS, Phi_phi0, Phi_r0

print('Done setting up loglike and prior_transform functions.')
print('Starting nested sampling...')

# Change to the desired directory
os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/sampling_test/')

# Add it to Python path
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/sampling_test/')
rstate = np.random.default_rng(7)
with dynesty.pool.Pool(16, loglike, prior_transform) as pool:
    dsampler = dynesty.DynamicNestedSampler(
        loglike,  
        prior_transform,
        ndim=10,
        bound='multi',
        sample='rwalk',
        rstate=rstate,
        nlive=500
    )
    # dsampler = dynesty.DynamicNestedSampler.restore('ns_likelihoodtest2.save', pool=pool)
    dsampler.run_nested(checkpoint_file='ns_likelihoodtest_wide.save')

print('Done nested sampling.')
print('Saving results to pickle...')
results = dsampler.results
with open('ns_likelihoodtest_wide.pkl', 'wb') as f:
    pickle.dump(results, f)