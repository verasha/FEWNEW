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

import GWfuncs
import loglikebasic
import modeselector
import parismc
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
gwf = GWfuncs.GravWaveAnalysis(T, dt)

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
param_true = [np.log10(m1), np.log10(m2), a, p0, e0]

# NOTE: change verbose argument for debugging
loglike_obj = loglikebasic.LogLike(params_star, waveform_gen_comb, gwf, M_init=5, verbose=False, waveform_gen_sep=waveform_gen_sep, noise_weighted=True)
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
    # same prior as paris_intrinsic.py
    logm1lim = [5.999851037559485, 6.000148962440515]
    logm2lim = [1.4770378504576676, 1.4772046589816572]
    alim = [0.6997315972442014, 0.7002684027557985]
    p0lim = [7.498519407697586, 7.501480592302414]
    e0lim = [0.3999359060325742, 0.40006409396742587]

    # 8 sigma
    # logm1lim = [5.999523320190352, 6.000476679809648]
    # logm2lim = [1.476854361081279, 1.4773881483580458]
    # alim = [0.6991411111814447, 0.7008588888185552]
    # p0lim = [7.495262104632274, 7.504737895367726]
    # e0lim = [0.3997948993042373, 0.40020510069576276]

    # 12 sigma
    # logm1lim = [5.9992849802855295, 6.0007150197144705]
    # logm2lim =  [1.4767209142620874, 1.4775215951772374]
    # alim =  [0.6987116667721671, 0.7012883332278328]
    # p0lim = [7.49289315694841, 7.50710684305159]
    # e0lim =  [0.3996923489563559, 0.40030765104364413]
    # 15 sigma
    # logm1lim = [5.999106225356911, 6.000893774643089]
    # logm2lim = [1.4766208291476937, 1.477621680291631]
    # alim = [0.698389583465209, 0.701610416534791]
    # p0lim = [7.491116446185513, 7.508883553814487]
    # e0lim = [0.39961543619544493, 0.4003845638045551]

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

def inverse_prior_transform(x):
    logm1lim = [5.999851037559485, 6.000148962440515]
    logm2lim = [1.4770378504576676, 1.4772046589816572]
    alim = [0.6997315972442014, 0.7002684027557985]
    p0lim = [7.498519407697586, 7.501480592302414]
    e0lim = [0.3999359060325742, 0.40006409396742587]

    u = np.zeros_like(x)
    
    # Inverse of: transformed = (max - min) * u + min
    # Solution: u = (transformed - min) / (max - min)
    
    u[:, 0] = (x[:, 0] - logm1lim[0]) / (logm1lim[1] - logm1lim[0])
    u[:, 1] = (x[:, 1] - logm2lim[0]) / (logm2lim[1] - logm2lim[0])
    u[:, 2] = (x[:, 2] - alim[0]) / (alim[1] - alim[0])
    u[:, 3] = (x[:, 3] - p0lim[0]) / (p0lim[1] - p0lim[0])
    u[:, 4] = (x[:, 4] - e0lim[0]) / (e0lim[1] - e0lim[0])
    
    return u

print('Done setting up log-likelihood and prior.')
print('Setting up ParisMC sampler...')
config = parismc.SamplerConfig(
    merge_confidence=0.9,          # Coverage prob → Mahalanobis merge radius R_m (higher is more permissive)
    alpha=10000,                    # Use recent samples for weighting. 
    trail_size=int(1e3),          # Maximum trials per iteration
    boundary_limiting=True,        # Enable boundary constraints
    use_beta=True,                # Use beta correction for boundaries
    integral_num=int(1e5),        # MC samples for beta estimation
    gamma=500,                    # Covariance update frequency NOTE: changed from 100
    exclude_scale_z=10,       # No exclusion based on weights
    use_pool=False,               # Set to True for multiprocessing
    n_pool=4                      # Number of processes (if use_pool=True)
)

print('Done setting up ParisMC sampler.')
print('Setting up initial covariance matrix...')
# Change to the desired directory
os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/sampling_test')

# Add it to Python path
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/sampling_test')

with open('cov_matrix_intrinsic_new.pkl', 'rb') as f:
    cov_matrix = pickle.load(f)

ndim = 5
n_seed = 1

init_cov_list = [cov_matrix for _ in range(n_seed)]

print('Done setting up initial covariance matrix.')

print('Initializing sampler...')
sampler = parismc.Sampler(
    ndim=ndim, 
    n_seed=n_seed,
    log_density_func=log_density,
    init_cov_list=init_cov_list,
    prior_transform=prior_transform,
    config=config
)
print('Done initializing sampler.')

print('Preparing LHS samples...')
# sampler.prepare_lhs_samples(lhs_num=int(1e5), batch_size=10)

# if using external LHS samples at true parameters
print('Using external LHS samples at true parameters...')
external_lhs_points = inverse_prior_transform(np.array([param_true]))
external_lhs_log_densities = log_density(prior_transform(external_lhs_points))
print('External LHS points:', external_lhs_points)
print('External LHS log densities:', external_lhs_log_densities)

print('Done preparing LHS samples.')

print('Running sampling...')
sampler.run_sampling(
    num_iterations=int(5e5), 
    savepath='./paris_intrinsic_ffunc_extlhs/',
    print_iter=100, # Print progress every n iterations
    external_lhs_points=external_lhs_points,
    external_lhs_log_densities=external_lhs_log_densities,

)
print('Done running sampling.')