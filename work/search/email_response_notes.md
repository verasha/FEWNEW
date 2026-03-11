# Response to Advisor Email - Mode Selection and Connection Plots

## Date
2025-12-15

## Email Summary

Advisor requests:
1. **Connection plots for f, X, and χ²** - break information into components to understand what's happening
2. **Extend n-indexed mode selection** - use ℓ=2, sum over all m, index by n=-1,0,1,2,... to get ~90% SNR
3. **Consider Fourier transform trick** - approximate maximization over time and phase shifts
4. **Investigate m=-2 dominance** - may indicate sign error (signal should be prograde)

## Current Status

### Mode Selection Investigation

#### Initial test: n = [-1, 2]
- **SNR captured**: 29.48% of total 5.28
- Individual contributions:
  - n=-1: 1.319 (24.96%)
  - n=0: 0.767 (14.53%)
  - n=1: 0.309 (5.85%)
  - n=2: 0.052 (0.99%)

#### Extended test: n = [-5, 2]
- **SNR captured**: 57.83%
- Strong plateau from n=-5 to n=-1 (~1.25-1.39 each)
- Sharp dropoff for positive n values

#### Wide test: n = [-15, 14]
- **SNR captured**: 70.26%
- Peak around n=-6: 1.474 (27.90% - strongest single mode)
- Range n=-8 to -1: All contribute ~0.75-1.47
- Outside this range: rapid decay

### Final Choice for Production Run

**Selected**: `n_vals = np.arange(-8, 3)` (n = -8 to 2)
- Captures ~68-70% of ℓ=2 SNR
- 11 n-groups total
- Each n-group sums over all m for ℓ=2

**Missing ~30% SNR likely from**:
- Higher ℓ modes (ℓ=3, 4, ...)
- Different physical basis may be needed

### Current Production Run

**File**: `intrinsic_ffunc_1mth_evenmoremodes.py`

**Parameters**:
```python
n_vals = np.arange(-8, 3)  # 11 n-groups
ell = 2  # quadrupole only
M_mode = None  # No SNR filtering
```

**Sampling config**:
- 500k iterations
- 100 seeds
- stop_dlogZ = 0.1
- Output: `./intrinsic_ffunc_1mth_evenmoremodes/`

## Connection Plot Analysis

### Key Finding: Disconnection is Sampling Artifact

**Theoretical connection** (direct likelihood evaluation):
- f-statistic, X, and χ² vary **smoothly** along line from max likelihood to true parameters
- No disconnection in actual likelihood surface

**Nearest neighbor connection** (sampler's searched points):
- Shows **drops to zero** between regions
- Artifact of sparse sampling, not physics

### Implication
- Likelihood surface is well-connected
- Sampling difficulty likely due to:
  - Missing 30% SNR making likelihood less informative
  - Need more modes for better parameter constraints

## Fourier Transform Trick (Time/Phase Marginalization)

### Current Implementation

**Phase marginalization**: Already implemented ✓
```python
# In GWfuncs.Xstat()
calc_inner_complex = self.inner(xf, hf, return_complex=True)
return self.xp.abs(calc_inner_complex) / calc_SNR
```
Taking `abs()` automatically maximizes over phase.

### Potential Addition: Time-shift Marginalization

**Concept**: Use FFT to maximize over time-of-arrival
```python
def Xstat_time_phase_marginalized(self, x, h):
    """
    Maximize over BOTH time shifts and phase using FFT correlation.
    """
    xf = self.freq_wave(x)
    hf = self.freq_wave(h)

    # Cross-correlation: S(f) = x̃(f) * h̃*(f)
    S_f = xf * self.xp.conj(hf)

    # IFFT gives correlation for ALL time shifts simultaneously
    S_t = self.xp.fft.irfft(S_f, axis=-1)

    # Maximum over all time shifts and phases
    max_correlation = self.xp.max(self.xp.abs(S_t))

    # Normalize
    calc_SNR = self.xp.sqrt(self.inner(hf, hf))
    return max_correlation / calc_SNR
```

**How it works**:
1. Frequency domain multiplication gives cross-correlation spectrum
2. Inverse FFT returns correlation at ALL time shifts τ
3. Taking max(|S(τ)|) finds optimal time shift and phase
4. This is O(N log N) instead of O(N²) for grid search

**When to use**:
- For intrinsic parameter estimation where exact time-of-arrival doesn't matter
- Makes likelihood more robust to time/phase errors
- Could improve sampler navigation

## Questions to Address

### 1. Why m=-2 dominates?
- Signal should be prograde (a=0.7 > 0)
- Expected to see mix of m values
- May indicate sign error somewhere
- **Action**: Check mode selection output and verify m distribution

### 2. What about missing 30% SNR?
- ℓ=2 with extended n only captures ~70%
- Need to test ℓ=3, 4, ... contributions
- **Action**: Modify mode selector to handle multiple ℓ values

### 3. Will more modes improve sampling?
- Current run with n=-8 to 2 (11 n-groups)
- Compare to previous n=-1 to 2 (4 n-groups)
- **Action**: Analyze if connection/sampling improves with more modes

## Next Steps

### Immediate
1. ✓ Running production sampling with n=-8 to 2
2. Monitor sampling progress and convergence
3. Create comparison plots: theoretical vs nearest-neighbor connections

### When run completes
1. Generate all three connection plots (f, X, χ²)
2. Compare sampling quality vs n=-1 to 2 run
3. Analyze if more modes reduced disconnection

### Follow-up investigations
1. Test ℓ=3 mode contributions
2. Investigate m=-2 dominance issue
3. Consider implementing time-shift marginalization if needed
4. Possibly combine multiple ℓ values to reach 90% SNR target

## Reference Files

- Mode selection: `/work/modeselectoralt.py`
- Likelihood: `/work/loglikealt.py`
- GW functions: `/work/GWfuncs.py`
- FFT reference: `fft_correlation_and_mode_decomposition_reference.md`
- Production run: `/work/search/intrinsic_ffunc_1mth_evenmoremodes.py`
- Connection plots notebook: `/work/search/intrinsic_ffunc_1mth_snr5_resume.ipynb`

## Technical Notes

### n-indexed mode basis
- For fixed ℓ and n, sum over all m
- Modes indexed as (ℓ, m, n) where:
  - ℓ = 2 (quadrupole)
  - m = -ℓ to +ℓ (summed)
  - n = harmonic index for radial motion
- Physical motivation: nearly orthogonal modes

### SNR Distribution Pattern
- Peak at n=-6 (strongest single contributor)
- Plateau from n=-8 to n=-1
- Rapid decay for |n| > 8
- Positive n much weaker than negative n

### Mode Count
- Each n-group has 2ℓ+1 = 5 modes (for ℓ=2)
- n=-8 to 2 → 11 groups × 5 modes = 55 individual (ℓ,m,n) modes
- But code sums each n-group, so likelihood sees 11 effective modes

## Code Snippets

### Mode selection call
```python
loglike_obj = loglikealt.LogLike(
    params_star,
    waveform_gen_comb,
    gwf,
    verbose=False,
    waveform_gen_sep=waveform_gen_sep,
    ell=ell,
    n_vals=n_vals,
    M_mode=None  # No SNR filtering, use all n-groups
)
```

### SNR diagnostic
```python
mode_waveforms = loglike_obj._generate_selected_waveforms(params_star, loglike_obj.flattened_modes)
mode_snrs = gwf.rhostat_modes([mode_waveforms[:, i] for i in range(mode_waveforms.shape[1])])
total_mode_snr = np.sqrt(np.sum(mode_snrs**2))
print(f'Percentage of total SNR captured: {total_mode_snr/data_snr * 100:.2f}%')
```