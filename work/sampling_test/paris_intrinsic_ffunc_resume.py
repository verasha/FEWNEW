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
import loglikebasic
import modeselector
import parismc
import pickle
import cupy as cp

# Configuration for resume
STATE_PATH = './sampling_test/paris_intrinsic_ffunc2/sampler_state.pkl'
MORE_ITERS = int(5e5)
OUT_DIR = './sampling_test/paris_intrinsic_ffunc2/'
PRINT_ITER = 100

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
    "force_backend": force_backend # Force GPU
}

# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "force_backend": force_backend,  # Force GPU
    # "assume_positive_m": True  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs_comb = {
    "force_backend": force_backend,  # Force GPU
    "pad_output": True,
}

sum_kwargs_sep = {
    "force_backend": force_backend,  # Force GPU
    "pad_output": True,
    "separate_modes": True,
}

print("Creating GenerateEMRIWaveform class...")
# Kerr eccentric flux
waveform_gen_comb = GenerateEMRIWaveform(
    FastKerrEccentricEquatorialFlux,
    frame='detector',
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs_comb,
    use_gpu=use_gpu
)

# Kerr eccentric flux
waveform_gen_sep = GenerateEMRIWaveform(
    FastKerrEccentricEquatorialFlux,
    frame='detector',
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs_sep,
    use_gpu=use_gpu
)


print('Done initializing waveform generator.')

print("Creating GravWaveAnalysis class...")
gwf = GWfuncs_backup2.GravWaveAnalysis(T, dt)

print("Initializing loglike class...")
# Source parameters
m1 = 1e6
m2 = 3e1
a = 0.7
p0 = 7.5
e0 = 0.4
## NOTE: BELOW THIS ALL ARE FIXED
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

params_star = (m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0)

# NOTE: change verbose argument for debugging
loglike_obj = loglikebasic.LogLike(params_star, waveform_gen_comb, gwf, M_init=5, verbose=False, waveform_gen_sep=waveform_gen_sep)
print('Done initializing loglike class.')

print("Setting up log_density and prior functions...")
def log_density(params):
    params = np.asarray(params)

    n_samples = params.shape[0]
    log_likes = np.zeros(n_samples)


    for i in range(n_samples):
        logm1, logm2, a, p0, e0 = params[i]
        m1 = 10**logm1
        m2 = 10**logm2

        loglike = loglike_obj(np.array([m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]))
        log_likes[i] = loglike

    return log_likes

def prior_transform(u):
    logm1lim = [5.999851037559485, 6.000148962440515]
    logm2lim = [1.4770378504576676, 1.4772046589816572]
    alim = [0.6997315972442014, 0.7002684027557985]
    p0lim = [7.498519407697586, 7.501480592302414]
    e0lim = [0.3999359060325742, 0.40006409396742587]

    transformed = np.zeros_like(u)

    # Uniform in log for masses

    # m1
    transformed[:, 0] = (logm1lim[1] - logm1lim[0]) * u[:, 0] + logm1lim[0]

    # m2
    transformed[:, 1] = (logm2lim[1] - logm2lim[0]) * u[:, 1] + logm2lim[0]

    # Linear in others

    # a
    transformed[:, 2] = (alim[1] - alim[0]) * u[:, 2] + alim[0]

    # p0
    transformed[:, 3] = (p0lim[1] - p0lim[0]) * u[:, 3] + p0lim[0]

    # e0
    transformed[:, 4] = (e0lim[1] - e0lim[0]) * u[:, 4] + e0lim[0]


    return transformed

print('Done setting up log-likelihood and prior.')


print(f"Loading sampler state from {STATE_PATH}...")
# Load saved sampler
sampler = parismc.Sampler.load_state(STATE_PATH)

# Rebind the functions for the loaded sampler
sampler.log_density_func_original = log_density
if hasattr(sampler, "prior_transform") and sampler.prior_transform is not None:
    sampler.prior_transform = prior_transform
    sampler.log_density_func = sampler.transformed_log_density_func
else:
    sampler.log_density_func = sampler.log_density_func_original

print("Loaded state successfully!")
print("Sampler State Summary:")
print("----------------------")
print(f"ndim: {sampler.ndim}")
print(f"n_proc: {sampler.n_seed}")
print(f"current_iter: {getattr(sampler, 'current_iter', None)}")
print(f"savepath: {getattr(sampler, 'savepath', '(unset)')}")

# Continue sampling
print(f"\nResuming sampling for {MORE_ITERS} more iterations...")
sampler.run_sampling(
    num_iterations=MORE_ITERS,
    savepath=OUT_DIR,
    print_iter=PRINT_ITER,
)

print("\nResume completed!")
print("Final Sampler State:")
print("--------------------")
print(f"current_iter: {getattr(sampler, 'current_iter', None)}")

# Quick analysis
try:
    samples, weights = sampler.get_samples_with_weights(flatten=True)
    ess = 1.0 / (weights ** 2).sum()
    wmean = (samples * weights[:, None]).sum(axis=0) / weights.sum()
    print("\nQuick Analysis:")
    print("---------------")
    print(f"Total samples: {len(samples)}")
    print(f"Effective sample size (ESS): {ess:.1f}")
    print(f"Weighted mean: {wmean}")
except Exception as e:
    print(f"Analysis skipped: {e}")

print("\nDone!")
