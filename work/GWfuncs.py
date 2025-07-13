import numpy as np
from SNR_tutorial_utils import LISA_Noise
from lisatools.sensitivity import *

class GravWaveAnalysis:
    """
    A module for GW data analysis that I've compiled.
    """

    # Physical constants
    Gpc = 3.0856775814913674e+25 # Gigaparsec in meters
    MRSUN_SI = 1476.6250615036158 # Mass-radius in SI units

    def __init__(self, N=None, dt=None):
        """
        Initialize the class with optional parameters.

        Parameters:
        dt (float): Time step for the data.
        fft_freqs (numpy.ndarray): Frequencies from FFT analysis.
        freq_mask (numpy.ndarray): Frequency mask for filtering.
        """
        self.N = N
        self.dt = dt
        self.fft_freqs = np.fft.rfftfreq(N, dt)  # Frequencies from FFT analysis
        # self.fft_freqs = fft_freqs
        # self.freq_mask = freq_mask

    

    def calc_power(self, teuk_modes, ylms, m0mask):
        """
        Calculate the power spectrum of the gravitational wave signal.

        Parameters:
        teuk_modes (numpy.ndarray): Teukolsky modes.
        ylms (numpy.ndarray): Spherical harmonics.
        m0mask (numpy.ndarray): Boolean mask where m!= 0.

        Returns:
        numpy.ndarray: Power summed over all trajectory points.
        """

        # Combine m>=0 and m<0 modes
        full_modes = np.concatenate([teuk_modes, np.conj(teuk_modes[:, m0mask])], axis=1)
        h_lmn = full_modes * ylms[np.newaxis, :] # (time, modes)
        power = np.abs(h_lmn)**2

        return np.sum(power, axis=0)  # Sum each mode over all trajectory points

    

    # def char_strain(self, hf):
    #     """
    #     Compute the characteristic strain.

    #     Parameters:
    #     hf (numpy.ndarray): Frequency domain waveform.

    #     Returns:
    #     numpy.ndarray: Characteristic strain.
    #     """
        
    #     # Compute the characteristic strain
    #     return 2*fft_freqs[freq_mask]*np.sqrt(np.abs(hf[0,freq_mask])**2+np.abs(hf[1,freq_mask])**2)

    def dist_factor(self, dist, mu):
        """
        Compute the distance factor for gravitational wave signals.

        Parameters:
        dist (float): Distance to the source in Gpc.

        Returns:
        float: Distance factor.
        """
        
        # Compute the distance factor
        return (dist * self.Gpc) / (mu * self.MRSUN_SI)

    def freq_wave(self, wave):
        """
        Compute the frequency domain representation of a waveform.

        Parameters:
        wave (numpy.ndarray): Time domain waveform.

        Returns:
        numpy.ndarray: Frequency domain waveform.
        """ 
        
        wave_c = np.vstack((wave.real, wave.imag))

        return np.fft.rfft(wave_c, axis=1)*self.dt

    def inner(self, h1f, h2f):
        """
        Compute the inner product of two gravitational waveforms.

        Parameters:
        h1f, h2f (numpy.ndarray): Frequency domain waveforms.

        Returns:
        float: Inner product of the two waveforms.
        """

        df = 1/(self.N*self.dt)  # Frequency resolution
        
        # Compute the inner product
        plus = np.conj(h1f[0,1:])@(h2f[0,1:]/LISA_Noise(self.fft_freqs[1:]))
        cross = np.conj(h1f[1,1:])@(h2f[1,1:]/LISA_Noise(self.fft_freqs[1:]))
        # plus = np.conj(h1f[0,1:])@(h2f[0,1:]/ get_sensitivity(self.fft_freqs[1:], sens_fn=LISASens, return_type="PSD"))
        # cross = np.conj(h1f[1,1:])@(h2f[1,1:]/get_sensitivity(self.fft_freqs[1:], sens_fn=LISASens, return_type="PSD"))

        return 4*df*np.real(plus+cross)

    def SNR(self, hf):
        """
        Compute the signal-to-noise ratio (SNR) for a gravitational wave signal.

        Parameters:
        hf (numpy.ndarray): Frequency domain waveform.

        Returns:
        float: Signal-to-noise ratio.
        """
        
        # Compute the SNR
        return np.sqrt(self.inner(hf,hf))

    def overlap(self, h1f, h2f):
        """
        Compute the overlap reduction function between two gravitational waveforms.

        Parameters:
        h1f, h2f (numpy.ndarray): Frequency domain waveforms.

        Returns:
        float: Overlap reduction function.
        """
        
        # Compute the overlap reduction function
        return self.inner(h1f,h2f)/(self.SNR(h1f)*self.SNR(h2f))
    