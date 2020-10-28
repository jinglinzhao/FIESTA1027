import numpy as np
import os
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy import stats
import copy
from FIESTA_functions import *

#==============================================================================
# Read the CCFs
#==============================================================================
FILE 		= sorted(glob.glob('./data/SOAP-fits/*.fits'))
N_file 		= len(FILE)

V_grid 		= (np.arange(401)-200)/10				# CCF Velocity grid [-20,20] km/s
RV_gauss 	= np.zeros(N_file)

CCF 		= np.zeros((V_grid.size, N_file))
eCCF 		= np.zeros((V_grid.size, N_file))
# Change to other S/N when necessary
SN 			= 10000

for n in range(N_file):
	hdulist  = fits.open(FILE[n])
	CCF[:,n] = (1 - hdulist[0].data)
	eCCF[:,n]= np.random.normal(0, (1-CCF[:,n])**0.5/SN)

#==============================================================================
# Feed CCFs into FIESTA
#==============================================================================
shift_spectrum, err_shift_spectrum, power_spectrum, err_power_spectrum, RV_gauss = FIESTA(V_grid, CCF+eCCF, eCCF)

# Convertion from km/s to m/s
shift_spectrum = shift_spectrum * 1000
err_shift_spectrum = err_shift_spectrum * 1000
power_spectrum = power_spectrum * 1000
err_power_spectrum = err_power_spectrum * 1000
RV_gauss = RV_gauss * 1000

RV_gauss = RV_gauss - np.mean(RV_gauss)

#==============================================================================
# Plots 
#==============================================================================
N_FIESTA_freq = shift_spectrum.shape[1]
# print('Because the number of frequencies calculated is large, only the \
# first {:d} frequencies are presented.'.format(N_FIESTA_freq))

fig, axes = plt.subplots(figsize=(12, 10))
for i in range(N_FIESTA_freq):
	ax = plt.subplot(N_FIESTA_freq,1,i+1)
	plt.plot(np.arange(N_file), shift_spectrum[:, i]-RV_gauss)
	plt.errorbar(np.arange(N_file), shift_spectrum[:, i]-RV_gauss, err_shift_spectrum[:, i], marker='.', ls='', alpha=0.5)
	plt.ylabel(r'RV$_{%d}$ [m/s]' %(i+1))
	if i != N_FIESTA_freq-1:
		ax.set_xticks([])
	else:
		plt.xlabel('time')
plt.savefig('time-series.png')
# plt.show()

#==============================================================================
# Write to file
#==============================================================================
if 0:
	for i in range(N_FIESTA_freq):
		np.savetxt('FIESTA'+str(i+1)+'.txt', shift_spectrum[:, i]-RV_gauss)
	for i in range(N_FIESTA_freq):
		np.savetxt('FIESTA'+str(i+1)+'_err.txt', err_shift_spectrum[:, i])

#==============================================================================
# Turn this on for simulation without noise
#==============================================================================
if 0:
	eCCF = np.zeros(CCF.shape)
	shift_spectrum, power_spectrum, RV_gauss = FIESTA(V_grid, CCF, eCCF)

	shift_spectrum = shift_spectrum * 1000
	power_spectrum = power_spectrum * 1000
	RV_gauss = RV_gauss * 1000
	RV_gauss = RV_gauss - np.mean(RV_gauss)

	N_FIESTA_freq = shift_spectrum.shape[1]

	fig, axes = plt.subplots(figsize=(12, 10))
	for i in range(N_FIESTA_freq):
		ax = plt.subplot(N_FIESTA_freq,1,i+1)
		plt.plot(np.arange(N_file), shift_spectrum[:, i]-RV_gauss)
		plt.plot(np.arange(N_file), shift_spectrum[:, i]-RV_gauss, marker='.', ls='', alpha=0.5)
		plt.ylabel(r'RV$_{%d}$ [m/s]' %(i+1))
		if i != N_FIESTA_freq-1:
			ax.set_xticks([])
		else:
			plt.xlabel('time')
	plt.savefig('time-series_no_noise.png')	

if 0:
	for i in range(N_FIESTA_freq):
		np.savetxt('FIESTA'+str(i+1)+'_no_noise.txt', shift_spectrum[:, i]-RV_gauss)
