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
FILE 		= sorted(glob.glob('./data/101501/ccfs/*.fits'))
N_file 		= len(FILE)

# get the template 
for n in range(N_file):
	hdulist     = fits.open(FILE[n])
	data         = hdulist[1].data
	if n == 0:
		max_flux =  max(data['ccf'])
		N_max = 0
	if max(data['ccf']) >= max_flux:
		max_flux = max(data['ccf'])
		N_tpl = n
		idx = (data['V_grid'] < 0.1e7) & (data['V_grid'] > -0.2e7)
		V_grid = data['V_grid'][idx] / 1e5

# Read and normalise the CCFs
for n in range(N_file):
	hdulist     = fits.open(FILE[n])
	data         = hdulist[1].data
	idx = (data['V_grid'] < 0.1e7) & (data['V_grid'] > -0.2e7)
	normalised_flux = (1 - data['ccf'] / np.mean(data['ccf']))[idx]
	try:
	    CCF
	except NameError:
		CCF = np.zeros((normalised_flux.size, N_file))
	try:
	    eCCF
	except NameError:
		eCCF = np.zeros((normalised_flux.size, N_file))
	CCF[:,n] = (1 - data['ccf'] / np.mean(data['ccf']))[idx]
	eCCF[:,n] = (data['e_ccf'] / np.mean(data['ccf']))[idx]
	# print(np.mean(data['e_ccf']))
	# print(np.mean(data['ccf'] / data['e_ccf']))

# plt.plot(V_grid, CCF)
# plt.ylim([0,0.5])
# plt.xlim([-25,15])
# plt.xlabel('V_grid [km/s]')
# plt.ylabel('Flux')
# plt.show()

#==============================================================================
# Feed CCFs into FIESTA
#==============================================================================
shift_spectrum, err_shift_spectrum, power_spectrum, err_power_spectrum, RV_gauss = FIESTA(V_grid, CCF, eCCF)

# Convertion from km/s to m/s
shift_spectrum 		= shift_spectrum * 1000
err_shift_spectrum 	= err_shift_spectrum * 1000
power_spectrum 		= power_spectrum * 1000
err_power_spectrum 	= err_power_spectrum * 1000
RV_gauss 			= RV_gauss * 1000
RV_gauss 			= RV_gauss - np.mean(RV_gauss)  # not used 

#==============================================================================
# Write to file
#==============================================================================
import pandas as pd
df 				= pd.read_csv('./data/101501/101501_activity.csv')
CCF_RV 			= df['CCF RV [m/s]'] - np.mean(df['CCF RV [m/s]'])
CBC_RV 			= df['CBC RV [m/s]'] - np.mean(df['CBC RV [m/s]'])
N_FIESTA_freq 	= shift_spectrum.shape[1]
if 1:
	for i in range(N_FIESTA_freq):
		np.savetxt('FIESTA'+str(i+1)+'.txt', shift_spectrum[:, i]-CCF_RV)
	for i in range(N_FIESTA_freq):
		np.savetxt('FIESTA'+str(i+1)+'_err.txt', err_shift_spectrum[:, i])

# write to CSV file


df_shift_spectrum = pd.DataFrame(data=np.hstack([shift_spectrum, err_shift_spectrum]),
                  columns=['x_0(xi_1)', 'x_0(xi_2)', 'x_0(xi_3)', 'x_0(xi_4)', 'x_0(xi_5)',
                 'err_x_0(xi_1)', 'err_x_0(xi_2)', 'err_x_0(xi_3)', 'err_x_0(xi_4)', 'err_x_0(xi_5)'])

df_shift_spectrum.to_csv('pennstate_fiesta1_results.csv')





#==============================================================================
# Plots 
#==============================================================================

#================#
# RV time series #
#================#
plt.rcParams.update({'font.size': 15})
plt.subplots(figsize=(12, 6))
plt.errorbar(df['Time [MJD-40000]'], CBC_RV, df['CBC RV Error [m/s]'], marker='o', ls='none', alpha= 0.5, label='CBC RV')
plt.errorbar(df['Time [MJD-40000]'], CCF_RV, df['CCF RV Error [m/s]'], marker='o', ls='none', alpha= 0.5, label='CCF RV')
plt.errorbar(df['Time [MJD-40000]'], RV_gauss, df['CCF RV Error [m/s]'], marker='o', ls='none', alpha= 0.5, label='RV_gauss')
plt.xlabel('Time [MJD-40000]')
plt.ylabel('RV [m/s]')
plt.legend()
plt.show()


def gaussian(x, amp, mu, sig, c):
    return amp * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) + c

CCF_RV = np.array(CCF_RV)
eCCF_RV = np.array(df['CCF RV Error [m/s]'])
time = np.array(df['Time [MJD-40000]'])
w_RV = np.zeros(time.shape)


def w(t):
	 return 1/eCCF_RV**2 * gaussian(abs(t-time), 1, 0, 1, 0)

# for i in range(len(time)):
w_RV = CCF_RV * w(time) / sum(w(time))

plt.plot(time, CCF_RV - w_RV, '.')
plt.show()



#===========================#
# FIESTA shifts time series #
#===========================#
fig, axes = plt.subplots(figsize=(12, 10))
for i in range(N_FIESTA_freq):
	ax = plt.subplot(N_FIESTA_freq,1,i+1)
	plt.errorbar(df['Time [MJD-40000]'], shift_spectrum[:, i] - CCF_RV, err_shift_spectrum[:, i], marker='o', ls='', alpha=0.5)
	plt.ylabel(r'RV$_{%d}$ [m/s]' %(i+1))
	if i != N_FIESTA_freq-1:
		ax.set_xticks([])
	else:
		plt.xlabel('MJD')
plt.show()	

#=============#
# Periodogram #
#=============#
from astropy.timeseries import LombScargle

time_span = (max(df['Time [MJD-40000]']) - min(df['Time [MJD-40000]']))
min_f   = 1/time_span
max_f   = 1 #1/(min(np.diff(df['Time [MJD-40000]'])))
spp     = 100  # spp=1000 will take a while; for quick results set spp = 10
xc 		= 17.1 # stellar rotational period?

def subplot_periodogram(x, y, yerr, panel_index, label, ticks=''):
	try:
		yerr
		frequency, power = LombScargle(x, y, yerr).autopower(minimum_frequency=min_f,
		                                                     maximum_frequency=max_f,
		                                                     samples_per_peak=spp)
	except NameError:
		frequency, power = LombScargle(x, y).autopower(minimum_frequency=min_f,
	                                                   maximum_frequency=max_f,
	                                                   samples_per_peak=spp)

	ax = plt.subplot(N_FIESTA_freq+6,1,panel_index)
	plt.plot(1/frequency, power, label=label, alpha=0.8)
	plt.axvline(x=xc, color='k', linestyle='--', alpha = 0.75)
	plt.xlim([1,time_span])
	plt.xscale('log')
	plt.legend()
	plt.ylabel("Power")
	if ticks == 'off':
		ax.set_xticks([])

fig, axes = plt.subplots(figsize=(12, 12))

subplot_periodogram(x = df['Time [MJD-40000]'], 
					y = df['CBC RV [m/s]'], 
					yerr = df['CBC RV Error [m/s]'],
					panel_index = 1, 
					label = 'CBC RV',
					ticks = 'off')

subplot_periodogram(x = df['Time [MJD-40000]'], 
					y = df['CCF RV [m/s]'], 
					yerr = df['CCF RV Error [m/s]'],
					panel_index = 2, 
					label = 'CCF RV',
					ticks = 'off')

for i in range(N_FIESTA_freq):

	subplot_periodogram(x = df['Time [MJD-40000]'], 
						y = shift_spectrum[:, i] - df['CCF RV [m/s]'], 
						yerr = err_shift_spectrum[:, i],
						panel_index = i+3, 
						label = r'$\xi$'+str(i+1),
						ticks = 'off')

subplot_periodogram(x = df['Time [MJD-40000]'], 
					y = df['H-alpha Emission'], 
					yerr = None,
					panel_index = N_FIESTA_freq+3,
					label = 'H-alpha Emission',
					ticks = 'off')

subplot_periodogram(x = df['Time [MJD-40000]'], 
					y = df['H-alpha Equiv. Width [A]'], 
					yerr = None,
					panel_index = N_FIESTA_freq+4,
					label = 'H-alpha Equiv. Width',
					ticks = 'off')

subplot_periodogram(x = df['Time [MJD-40000]'], 
					y = df['CCF FWHM [m/s]'], 
					yerr = None,
					panel_index = N_FIESTA_freq+5,
					label = 'CCF FWHM',
					ticks = 'off')

subplot_periodogram(x = df['Time [MJD-40000]'], 
					y = df['BIS [m/s]'], 
					yerr = None,
					panel_index = N_FIESTA_freq+6,
					label = 'BIS')
plt.xlabel("Period [d]")
plt.savefig('Periodogram.png')
plt.show()

