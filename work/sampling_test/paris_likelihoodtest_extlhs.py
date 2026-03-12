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

import localgit.FEWNEW.work.GWfuncs_backup2 as GWfuncs_backup2
import loglike
import modeselector
import parismc
# import gc
# import pickle
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

#Generating data (true)

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
param_true = [np.log10(m1), np.log10(m2), a, p0, e0, dist, np.cos(qS), phiS, Phi_phi0, Phi_r0]


print('Setting up GWFuncs...')
gwf = GWfuncs_backup2.GravWaveAnalysis(T, dt)
print('Done setting up GWFuncs.')

print('Setting up log-likelihood and prior...')
def loglike(params):

    params = np.asarray(params)

    n_samples = params.shape[0]
    log_likes = np.zeros(n_samples)

    for i in range(n_samples):
        logm1, logm2, a, p0, e0, dist, cosqS, phiS, Phi_phi0, Phi_r0 = params[i]
        m1 = 10**logm1
        m2 = 10**logm2
        qS = np.arccos(cosqS)
        phiK = phiS + np.pi/3

        htemp = waveform_gen(m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
                            Phi_phi0, Phi_theta0, Phi_r0, T=T, dt=dt)
        

        res = data - htemp
        res_f = gwf.freq_wave(res)
        inner_res = gwf.inner(res_f, res_f)
        log_likes[i] = -0.5 * inner_res

    return log_likes

    
def prior_transform(u):


    # WIDER using diag of cov
    # 3 sigma
    logm1lim = [5.999755966003094, 6.000244033996906]
    logm2lim = [1.4769870513061485, 1.4772554581331763]
    alim = [0.6995637359288023, 0.7004362640711976]
    p0lim = [7.497572969128289, 7.502427030871711]
    e0lim = [0.39989407441630453, 0.4001059255836955]
    distlim = [0.4671238045922167, 0.5328761954077833]
    cosqSlim = [0.8175373164938913, 0.9376278072868542]
    phiSlim = [0.8907980491383655, 1.1092019508616344]
    Phiphilim = [0.2971011746225549, 0.5028988253774451]
    Phirlim = [0.4374942669461721, 0.5625057330538279]



    transformed = np.zeros_like(u)

    # Uniform in log for masses

    # m1
    transformed[:, 0] = (logm1lim[1] - logm1lim[0]) * u[:, 0] + logm1lim[0]

    # m2
    transformed[:, 1] = (logm2lim[1] - logm2lim[0]) * u[:, 1] + logm2lim[0]

    # Uniform for other parameters

    # a
    transformed[:, 2] = (alim[1] - alim[0]) * u[:, 2] + alim[0]

    # p0
    transformed[:, 3] = (p0lim[1] - p0lim[0]) * u[:, 3] + p0lim[0] 

    # e0
    transformed[:, 4] = (e0lim[1] - e0lim[0]) * u[:, 4] + e0lim[0]

    # dist 
    transformed[:, 5] = (distlim[1] - distlim[0]) * u[:, 5] + distlim[0]

    # qS
    transformed[:, 6] = (cosqSlim[1] - cosqSlim[0]) * u[:, 6] + cosqSlim[0]

    # phiS
    transformed[:, 7] = (phiSlim[1] - phiSlim[0]) * u[:, 7] + phiSlim[0]

    # Phi_phi0
    transformed[:, 8] = (Phiphilim[1] - Phiphilim[0]) * u[:, 8] + Phiphilim[0]

    # Phi_r0
    transformed[:, 9] = (Phirlim[1] - Phirlim[0]) * u[:, 9] + Phirlim[0]

    
    return transformed


def inverse_prior_transform(x):
    # WIDER using diag of cov
    logm1lim = [5.999755966003094, 6.000244033996906]
    logm2lim = [1.4769870513061485, 1.4772554581331763]
    alim = [0.6995637359288023, 0.7004362640711976]
    p0lim = [7.497572969128289, 7.502427030871711]
    e0lim = [0.39989407441630453, 0.4001059255836955]
    distlim = [0.4671238045922167, 0.5328761954077833]
    cosqSlim = [0.8175373164938913, 0.9376278072868542]
    phiSlim = [0.8907980491383655, 1.1092019508616344]
    Phiphilim = [0.2971011746225549, 0.5028988253774451]
    Phirlim = [0.4374942669461721, 0.5625057330538279]

    u = np.zeros_like(x)
    
    # Inverse of: transformed = (max - min) * u + min
    # Solution: u = (transformed - min) / (max - min)
    
    u[:, 0] = (x[:, 0] - logm1lim[0]) / (logm1lim[1] - logm1lim[0])
    u[:, 1] = (x[:, 1] - logm2lim[0]) / (logm2lim[1] - logm2lim[0])
    u[:, 2] = (x[:, 2] - alim[0]) / (alim[1] - alim[0])
    u[:, 3] = (x[:, 3] - p0lim[0]) / (p0lim[1] - p0lim[0])
    u[:, 4] = (x[:, 4] - e0lim[0]) / (e0lim[1] - e0lim[0])
    u[:, 5] = (x[:, 5] - distlim[0]) / (distlim[1] - distlim[0])
    u[:, 6] = (x[:, 6] - cosqSlim[0]) / (cosqSlim[1] - cosqSlim[0])
    u[:, 7] = (x[:, 7] - phiSlim[0]) / (phiSlim[1] - phiSlim[0])
    u[:, 8] = (x[:, 8] - Phiphilim[0]) / (Phiphilim[1] - Phiphilim[0])
    u[:, 9] = (x[:, 9] - Phirlim[0]) / (Phirlim[1] - Phirlim[0])
    
    return u

print('Done setting up log-likelihood and prior.')
print('Setting up ParisMC sampler...')
config = parismc.SamplerConfig(
    merge_confidence=0.9,          # Coverage prob → Mahalanobis merge radius R_m (higher is more permissive)
    alpha=10000,                    # Use recent samples for weighting. NOTE: can change to 10k
    trail_size=int(1e3),          # Maximum trials per iteration
    boundary_limiting=True,        # Enable boundary constraints
    use_beta=True,                # Use beta correction for boundaries
    integral_num=int(1e5),        # MC samples for beta estimation
    gamma=100,                    # Covariance update frequency
    exclude_scale_z=10,       # No exclusion based on weights
    use_pool=False,               # Set to True for multiprocessing
    n_pool=4                      # Number of processes (if use_pool=True)
)
print('Done setting up ParisMC sampler.')
print('Setting up initial covariance matrix...')
#Changing directory and loading covariance matrix
os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/sampling_test')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/sampling_test')

import pickle
with open('cov_matrix_new.pkl', 'rb') as f:
    cov_matrix = pickle.load(f)

ndim = 10
n_seed = 1 # Number of initial walkers/chains. can just change to 1


init_cov_list = [cov_matrix for _ in range(n_seed)]

print('Done setting up initial covariance matrix.')
print('Initializing sampler...')
sampler = parismc.Sampler(
    ndim=ndim, 
    n_seed=n_seed,
    log_density_func=loglike,
    init_cov_list=init_cov_list,
    prior_transform=prior_transform,
    config=config
)
print('Done initializing sampler.')
print('Preparing LHS samples...')
# sampler.prepare_lhs_samples(lhs_num=int(1e5), batch_size=100) #NOTE: change to 10-100

# if using external LHS samples at true parameters
print('Using external LHS samples at true parameters...')
external_lhs_points = inverse_prior_transform(np.array([param_true]))
external_lhs_log_densities = loglike(prior_transform(external_lhs_points))
print('External LHS points:', external_lhs_points)
print('External LHS log densities:', external_lhs_log_densities)

print('Done preparing LHS samples.')
print('Running sampling...')
sampler.run_sampling(
    num_iterations=int(1e5),  #NOTE: 
    savepath='./paris_likelihoodtest_extlhs/',
    print_iter=100, # Print progress every n iterations
    external_lhs_points=external_lhs_points,
    external_lhs_log_densities=external_lhs_log_densities,

)
print('Done running sampling.')