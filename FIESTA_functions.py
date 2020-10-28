import numpy as np
from scipy.optimize import curve_fit
import copy

#==============================================================================
# 
# General functions
# 
#==============================================================================

# Gaussian function
def gaussian(x, amp, mu, sig, c):
    return amp * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) + c

# Fourier transform (FFT)
# https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html#numpy.fft.rfft
# freq returns only the positive frequencies 
def FT(signal, spacing):
	n 			= signal.size
	fourier 	= np.fft.rfft(signal, n)
	freq 		= np.fft.rfftfreq(n, d=spacing)
	power 		= np.abs(fourier)
	phase 		= np.angle(fourier)
	return [power, phase, freq]


# Fourier transform (FFT)
# https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html#numpy.fft.fft
# freq returns both positive and negative frequencies 
def FT2(signal, spacing):
	n 			= signal.size
	fourier 	= np.fft.fft(signal, n)
	freq 		= np.fft.fftfreq(n, d=spacing)	
	power 		= np.abs(fourier)
	phase 		= np.angle(fourier)
	return [fourier, power, phase, freq]


def unwrap(array):
# An individual phase ranges within (-np.pi, np.pi)
# The difference of two phases ranges within (-2*np.pi, 2*np.pi)
# Adding a phase of multiples of 2*np.pi is effectively the same phase
# This function wraps the phase difference such that it lies within (-np.pi, np.pi)
	for i in np.arange(len(array)):
		array[i] = array[i] - int(array[i]/np.pi) * 2 * np.pi
	return array


#==============================================================================
# 
# FourIEr phase SpecTrum Analysis (FIESTA)
# 
#==============================================================================

def FIESTA(V_grid, CCF, eCCF):
	N_file = CCF.shape[1]
	spacing = np.diff(V_grid)[0]

	# construct a template
	if ~np.all((eCCF == 0)):
		tpl_CCF = np.sum(CCF/eCCF**2, axis=1) / np.sum(1/eCCF**2, axis=1)
	else:
		tpl_CCF = CCF[:,0]

	# Choose the "interesting" part of V_grid for analysis.
	# The following chooses the range up to 5-sigma.
	popt, pcov 	= curve_fit(gaussian, V_grid, tpl_CCF, p0=[0.5, (max(V_grid)+min(V_grid))/2, 1, 0])
	sigma 		= popt[2]
	V_centre 	= popt[1]
	V_min 		= V_centre - 5*sigma
	V_max 		= V_centre + 5*sigma

	# reshape all input spectra 
	idx 	= (V_grid>V_min) & (V_grid<V_max)
	tpl_CCF = tpl_CCF[idx]
	V_grid 	= V_grid[idx]
	CCF 	= CCF[idx,:]
	if ~np.all((eCCF == 0)):
		eCCF 	= eCCF[idx,:]

	# Power spectrum
	_, power_tpl, phase_tpl, freq = FT2(tpl_CCF, spacing)

	# If no noise is present in the input spectra, only use the first 5 positive frequency modes
	if np.all((eCCF == 0)):
		freq_max = freq[6]
	else:
		# Determine the appropriate frequency range. 
		'''
		Inverse FT with frequencies within [-freq_max, freq_max] 
		should have residules no smaller than the noise level for individual CCF.
		=> Inverse FT with frequencies within [-freq_max, freq_max
		should have residules no smaller than the noise level for CCF with lowest S/N. 
		'''

		## First, locate the CCF with lowerst S/N. 
		mean_eCCF = np.mean(eCCF, axis=0)
		idx_noisest_CCF = (mean_eCCF == max(mean_eCCF))

		## Then, determine the appropriate frequency range for the lowest S/N CCF .
		for freq_max in freq:
			idx_freq = abs(freq) <= freq_max
			ft, power, phase, freq = FT2(CCF[:,idx_noisest_CCF].flatten(), spacing)
			pseudo_ft = copy.copy(ft)
			pseudo_ft[~idx_freq] = 0
			pseudo_ift = np.fft.ifft(pseudo_ft)
			residual = abs(pseudo_ift - CCF[:,idx_noisest_CCF].flatten())
			if np.median(residual) < np.median(eCCF[:,idx_noisest_CCF]):
				freq_max = freq_max
				break
	# Change the following 6 if necessary. 
	# In this case, it only returns the first 5 positive frequencies.
	if freq_max > freq[6]:
		freq_max = freq[6]
	freq_FIESTA = freq[(freq>0) & (freq<freq_max)]
	print('The frequencies used for the FIESTA analysis are')
	for fre in enumerate(freq_FIESTA, start=1):
		print(fre)
	print()

	if ~np.all((eCCF == 0)):
		# Estimate noise in the Fourier space by simulating noise for individual CCF
		def noise_for_power_phase(signal, noise, N):
			power_tpl, phase_tpl, freq = FT(signal, spacing)
			idx = (freq < freq_max) & (freq > 0)
			noise_shift_spectrum = np.zeros((N, len(freq_FIESTA)))
			noise_power_spectrum = np.zeros((N, len(freq_FIESTA)))
			for n in range(N):
				signal_noise = np.random.normal(signal, abs(noise))
				power, phase, _ = FT(signal_noise, spacing)
				diff_phase = unwrap(phase[idx] - phase_tpl[idx])
				noise_shift_spectrum[n, :] = -diff_phase / (2*np.pi*freq_FIESTA)
				noise_power_spectrum[n, :] = (power-power_tpl)[idx]
			return noise_shift_spectrum, noise_power_spectrum

	# FIESTA power spectrum, shift spectrum (relative to the template) and noise estimate 
	for n in range(N_file):

		## power spectrum and shift spectrum
		try:
		    power_spectrum
		except NameError:
			power_spectrum = np.zeros((N_file, freq_FIESTA.size))
		try:
		    shift_spectrum
		except NameError:
			shift_spectrum = np.zeros((N_file, freq_FIESTA.size))

		_, power, phase, _ = FT2(CCF[:,n], spacing)
		diff_phase = unwrap(phase - phase_tpl)
		idx = (freq<freq_max) & (freq>0)	
		power_spectrum[n,:] = power[idx]
		shift_spectrum[n,:] = -diff_phase[idx] / (2*np.pi*freq_FIESTA)

		## RV measured as centroid of a Gaussian fit.
		try:
		    RV_gauss
		except NameError:
			RV_gauss = np.zeros((N_file))
		popt, pcov 	= curve_fit(gaussian, V_grid, CCF[:,n])
		RV_gauss[n] = popt[1]

		## FIESTA error estimated with 1000 realisations of simulating photon noise.
		if ~np.all((eCCF == 0)):					
			try:
			    err_shift_spectrum
			except NameError:
				err_shift_spectrum = np.zeros((N_file, freq_FIESTA.size))
			try:
			    err_power_spectrum
			except NameError:
				err_power_spectrum = np.zeros((N_file, freq_FIESTA.size))		

			noise_shift_spectrum, noise_power_spectrum = noise_for_power_phase(CCF[:,n], eCCF[:,n], 1000)
			err_shift_spectrum[n,:] = np.std(noise_shift_spectrum, axis=0)
			err_power_spectrum[n,:] = np.std(noise_power_spectrum, axis=0)

	# If noise is present in the input spectra (as is normally the case),
	# return the FIESTA outputs with error estimates.
	# Otherwise (e.g. for quicker results or simulations), 
	# return the FIESTA outputs without error estimates.
	if ~np.all((eCCF == 0)):
		return shift_spectrum, err_shift_spectrum, power_spectrum, err_power_spectrum, RV_gauss
	else: 
		return shift_spectrum, power_spectrum, RV_gauss