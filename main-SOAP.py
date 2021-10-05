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
# from FIESTA_functions import *
import sys
sys.path.append("../")
from FIESTA_functions import *
#==============================================================================
# Read the CCFs
#==============================================================================
FILE 		= sorted(glob.glob('./data/SOAP-fits/*.fits'))
N_file 		= len(FILE)

V_grid0 	= (np.arange(401)-200)/10				# CCF Velocity grid [-20,20] km/s

# V_grid0 	= ((V_grid0[:-1] + V_grid0[1:])/2)[::2]

# V_grid0 	= V_grid0[:-2]
# V_grid0 	= ((V_grid0[0::3] + V_grid0[1::3] + V_grid0[2::3])/3)

# V_grid0 	= V_grid0[:-1]
# V_grid0 	= ((V_grid0[0::4] + V_grid0[1::4] + V_grid0[2::4] + V_grid0[3::4])/4)

# V_grid0 	= V_grid0[:-1]
# V_grid0 	= ((V_grid0[0::8] + V_grid0[1::8] + V_grid0[2::8] + V_grid0[3::8] + V_grid0[4::8] + V_grid0[5::8] + V_grid0[6::8] + V_grid0[7::8]) / 8)

idx 		= (-10 < V_grid0) & (V_grid0 < 10)
V_grid 		= V_grid0[idx]
RV_gauss 	= np.zeros(N_file)

CCF 		= np.zeros((V_grid.size, N_file))
CCF0 		= np.zeros((V_grid0.size, N_file))
eCCF 		= np.zeros((V_grid.size, N_file))
# Change to other S/N when necessary
SN 			= 10000000

if 0:
	for n in range(N_file):
		hdulist  = fits.open(FILE[n])
		temp = (1 - hdulist[0].data)

		CCF0[:, n] = ((temp[:-1] + temp[1:])/2)[::2]

		# temp 	= temp[:-2]
		# CCF0[:, n] = ((temp[0::3] + temp[1::3] + temp[2::3]) / 3)

		# temp 	= temp[:-1]
		# CCF0[:, n] = ((temp[0::4] + temp[1::4] + temp[2::4] + temp[3::4]) / 4)

		# temp 	= temp[:-1]
		# CCF0[:, n] = (temp[0::8] + temp[1::8] + temp[2::8] + temp[3::8] + temp[4::8] + temp[5::8] + temp[6::8] + temp[7::8])/8

		CCF[:,n] = CCF0[:, n][idx]
		eCCF[:,n]= np.random.normal(0, (1-CCF[:,n])**0.5/SN/2**0.5)

for n in range(N_file):
	hdulist  = fits.open(FILE[n])
	CCF0[:, n] = (1 - hdulist[0].data)
	CCF[:,n] = (1 - hdulist[0].data)[idx]
	# eCCF[:,n]= np.random.normal(0, (1-CCF[:,n])**0.5/SN)

# #
# plt.plot(V_grid, CCF, '.')
# plt.savefig('CCF8.png')
# plt.show()
#==============================================================================
# Feed CCFs into FIESTA
#==============================================================================
# shift_spectrum, err_shift_spectrum, power_spectrum, err_power_spectrum, RV_gauss = FIESTA(V_grid, CCF, eCCF, template=CCF[:,0])
shift_spectrum, _, power_spectrum, _, RV_gauss = FIESTA(V_grid, CCF, eCCF, template=CCF[:,0], k_max=50)

# Convertion from km/s to m/s
shift_spectrum = shift_spectrum * 1000
# err_shift_spectrum = err_shift_spectrum * 1000
RV_gauss = RV_gauss * 1000

RV_gauss = RV_gauss - RV_gauss[0]

idx = (RV_gauss>0.3) & (RV_gauss < 0.4)
#
# plt.plot(power_spectrum[0,0:5], '.')
# plt.show()



#===================#
# begin publication #
#===================#
plt.rcParams.update({'font.size': 18})
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

rotation_phase = np.arange(N_file)/100
alpha = 0.7
fig, axes = plt.subplots(figsize=(10, 6))
for i in range(5):
	plt.plot(rotation_phase, shift_spectrum[i,:] - shift_spectrum[i,0], color=colors[i], lw=4-i*0.7, marker='o', alpha=1-i*0.2, label=r'$k={}$'.format(i+1))
plt.xlim([0.2,0.8])
plt.xlabel('Rotation phase')
plt.ylabel(r'$RV_{FT, k}$ [m/s]')
plt.legend()
plt.savefig('FIESTA_II_insight_RV_k.png')
plt.show()

rv_weighted = np.zeros(shift_spectrum.shape[1])
for i in range(shift_spectrum.shape[1]):
	rv_weighted[i] = sum(shift_spectrum[:,i] * power_spectrum[:,i]) / np.sum(power_spectrum[:,i])

fig, axes = plt.subplots(figsize=(10, 6))
plt.plot(rotation_phase, rv_weighted-rv_weighted[0], marker='o', lw=3, color='black', alpha=1, label='Weighted mean')
# plt.scatter(rotation_phase, rv_weighted-rv_weighted[0], marker='s', s=100, label='Weighted mean', alpha=0.5)
# plt.plot(rotation_phase, RV_gauss-RV_gauss[0], marker='o', alpha=alpha, label=r'$RV_{gaussian}$')
plt.scatter(rotation_phase, RV_gauss-RV_gauss[0], marker='X', s=150, color='black', lw=1, label=r'$RV_{gaussian}$', alpha=0.5)
plt.xlabel('Rotation phase')
plt.ylabel('RV [m/s]')
plt.xlim([0.2,0.8])
plt.legend(loc=1)
plt.savefig('FIESTA_II_insight_weighted_mean.png')
plt.show()

#--------------------#
# A challenging test #
#--------------------#
shift_spectrum, _, power_spectrum, _, RV_gauss = FIESTA(V_grid, CCF, eCCF, template=CCF[:,0])
# Convertion from km/s to m/s
shift_spectrum = shift_spectrum * 1000
# err_shift_spectrum = err_shift_spectrum * 1000
RV_gauss = RV_gauss * 1000
RV_gauss = RV_gauss - RV_gauss[0]

# Singal  #
def gaussian2(x, amp, mu, sig, c):
    return amp * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) + c

f = interp1d(V_grid0, CCF0[:,0], kind='cubic')
ccf_shift = f(V_grid - RV_gauss[51]/1000)  # use interpolation function returned by `interp1d`
ADD_NOISE = False
if ADD_NOISE == True:
	ccf_shift = np.random.normal(ccf_shift, (1 - ccf_shift) ** 0.5 / 10000)
popt_shift, pcov = curve_fit(gaussian2, V_grid, ccf_shift)
popt, pcov = curve_fit(gaussian2, V_grid, CCF[:,0])
RV_shift = popt_shift[1] - popt[1]
print(RV_shift*1000)



CCF_plot = np.zeros((CCF.shape[0],3))
CCF_plot[:,0] = CCF[:,0]
CCF_plot[:,1] = ccf_shift
CCF_plot[:,2] = CCF[:,51]
eCCF_plot = np.zeros(CCF_plot.shape)
SN 			= 100000000
for n in range(3):
	eCCF_plot[:,n]= np.random.normal(0, (1-CCF_plot[:,n])**0.5/SN)
# CCF_plot 	= np.hstack((CCF_plot, CCF))
# eCCF_plot 	= np.hstack((eCCF_plot, eCCF))

shift_spectrum, err_shift_spectrum, power_spectrum, err_power_spectrum, RV_gauss = FIESTA(V_grid, CCF_plot, eCCF_plot, template=CCF_plot[:,0], k_max=6)

# SN 			= 100000
# eCCF_plot_noise = np.zeros(CCF_plot.shape)
# for n in range(3):
# 	eCCF_plot_noise[:,n]= np.random.normal(0, (1-CCF_plot[:,n])**0.5/SN)
# shift_spectrum2, err_shift_spectrum2, power_spectrum2, err_power_spectrum2, RV_gauss2 = FIESTA(V_grid, CCF_plot, eCCF_plot_noise, template=CCF[:,0], k_max = 6)


spacing = np.diff(V_grid)[0]

power0, phase0, freq = FT(CCF[:,0], spacing)	# template CCF
power1, phase1, freq = FT(CCF_plot[:,1], spacing)	# shifted CCF 
power2, phase2, freq = FT(CCF_plot[:,2], spacing)	# deformed CCF
idx = np.arange(k_max)

diff_phase = wrap(phase2-phase0)
diff_phase_shift = wrap(phase1-phase0)

x = V_grid

plt.rcParams.update({'font.size': 16})
fig = plt.figure(figsize=(18, 11))
# fig, axes = plt.subplots(figsize=(18, 11))
plt.subplots_adjust(hspace=0.4, wspace=0.5)  # the amount of width and height reserved for blank space between subplots
lw=3

from matplotlib.ticker import FormatStrFormatter
def mode_number(x):
	return x / np.diff(freq)[0]
inverse = mode_number

plt.subplot(231)
plt.plot(x, CCF_plot[:,2], 'r', alpha=alpha, lw=lw)
plt.plot(x, CCF_plot[:,1], 'b--', alpha=alpha, lw=lw)
# plt.plot(x, CCF_plot[:,1]+eCCF_plot_noise[:,1], 'b.', alpha=0.3)
# plt.plot(x, CCF_plot[:,2]+eCCF_plot_noise[:,2], 'r.', alpha=0.3)
# plt.title('Signal (CCF)')
plt.xlabel(r'$v$ [km/s]')
plt.ylabel('CCF (inverted)')
plt.grid(True)

# Singal deformation #
plt.subplot(234)
plt.plot(x, CCF_plot[:,2] - CCF_plot[:,0], 'r', alpha=alpha, lw=lw)
plt.plot(x, CCF_plot[:,1] - CCF_plot[:,0], 'b--', alpha=alpha, lw=lw)
# plt.plot(x, CCF_plot[:,1] - CCF_plot[:,0] + eCCF_plot_noise[:,1] - eCCF_plot_noise[:,0], 'b-', alpha=0.3)
# plt.plot(x, CCF_plot[:,2] - CCF_plot[:,0] + eCCF_plot_noise[:,2] - eCCF_plot_noise[:,0], 'r-', alpha=0.3)
plt.xlabel(r'$v$ [km/s]')
plt.ylabel(r'$\Delta$CCF')
plt.grid(True)


# power spectrum #
ax = fig.add_subplot(232)
ax.plot(freq[idx], power2[idx], 'rs-', alpha=alpha, lw=lw)
ax.plot(freq[idx], power1[idx], 'bo--', alpha=alpha, lw=lw)
# plt.errorbar(freq[idx], power2[idx], err_power_spectrum2[2,idx], c='r', marker='s', ms=5, alpha=0.8)
# plt.errorbar(freq[idx], power1[idx], err_power_spectrum2[1,idx], c='b', marker='o', ms=5, alpha=0.8)
# plt.errorbar(freq[idx], diff_phase_shift[idx], err_shift_spectrum2[1,idx] * (2 * np.pi * freq[idx]),  marker='o', ms=5, alpha=0.8)
# plt.errorbar(freq[idx], diff_phase[idx], err_shift_spectrum2[1,idx] * (2 * np.pi * freq[idx]), c='r', marker='s', ms=5, alpha=0.8)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_xticks(freq[idx], minor=False)
ax.set_xlabel(r'$\xi$ [s/km]')
ax.set_ylabel(r'$A_k$')
ax.grid(b=True, which='both')
secax = ax.secondary_xaxis('top', functions=(mode_number, inverse))	
secax.set_xticks(idx, minor=False)
secax.set_xlabel(r'$k$')


# power spectrum difference#

ax = fig.add_subplot(233)
ax.plot(freq[idx], power2[idx]-power0[idx], 'rs-', alpha=alpha, lw=lw)
ax.plot(freq[idx], power1[idx]-power0[idx], 'bo--', alpha=alpha, lw=lw)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_xticks(freq[idx], minor=False)
ax.set_xlabel(r'$\xi$ [s/km]')
ax.set_ylabel(r'$\Delta A_k$')
ax.grid(b=True, which='both')
secax = ax.secondary_xaxis('top', functions=(mode_number, inverse))	
secax.set_xticks(idx, minor=False)
secax.set_xlabel(r'$k$')




ax = fig.add_subplot(235)

ax.plot(freq[idx], diff_phase[idx], 'rs-', alpha=alpha, lw=lw)
ax.plot(freq[idx], diff_phase_shift[idx], 'bo--', alpha=alpha, lw=lw)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_xticks(freq[idx], minor=False)
ax.set_xlabel(r'$\xi$ [s/km]')
ax.set_ylabel(r'$\Delta \phi_k$ [radian]')
ax.grid(b=True, which='both')
secax = ax.secondary_xaxis('top', functions=(mode_number, inverse))	
secax.set_xticks(idx, minor=False)
secax.set_xlabel(r'$k$')



# shift spectrum #
ax = fig.add_subplot(236)

rv = np.zeros(len(diff_phase))
rv[1:] = - diff_phase[1:] / (2 * np.pi * freq[1:])
rv[0] = rv[1]

rv_shift = np.zeros(len(diff_phase_shift))
rv_shift[1:] = - diff_phase_shift[1:] / (2 * np.pi * freq[1:])
# rv_shift[0] = rv_shift[1]

# idx = np.arange(shift_spectrum2.shape[1]-1)
ax.plot(freq[1:6], rv[1:6] * 1000, 'rs-', alpha=alpha, lw=lw)
ax.plot(freq[1:6], rv_shift[1:6] * 1000, 'bo--', alpha=alpha, lw=lw)
# plt.errorbar(freq[idx+1], rv[idx+1]*1000, err_shift_spectrum2[2,idx+1]*1000, c='r', marker='s', ms=5, alpha=0.8)
# plt.errorbar(freq[idx+1], rv_shift[idx+1]*1000, err_shift_spectrum2[1,idx+1]*1000, c='b', marker='o', ms=5, alpha=0.8)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_xticks(freq[idx], minor=False)
ax.set_xlabel(r'$\xi$ [s/km]')
ax.set_ylabel(r'$\Delta RV_k$ [m/s]')
ax.grid(b=True, which='both')
secax = ax.secondary_xaxis('top', functions=(mode_number, inverse))	
secax.set_xticks(idx, minor=False)
secax.set_xlabel(r'$k$')
if ADD_NOISE == False:
	plt.savefig('comparison_noise-free.png')
else:
	plt.savefig('comparison_noise-added.png')
plt.show()


#=================#
# end publication #
#=================#


shift_function = np.zeros(shift_spectrum.shape)
for i in range(shift_spectrum.shape[1]):
	shift_function[:, i] = shift_spectrum[:, i] - RV_gauss

fig, axes = plt.subplots(figsize=(12, 10))
for i in range(N_FIESTA_freq):
	ax = plt.subplot(N_FIESTA_freq,1,i+1)
	if i ==0:
		plt.title('FIESTA RV – OBS RV')
	plt.plot(np.arange(N_file)/100, shift_function[:, i])
	plt.errorbar(np.arange(N_file)/100, shift_function[:, i], err_shift_spectrum[:, i], marker='.', ls='', alpha=0.5)
	plt.ylabel(r'mode ${%d}$ [m/s]' %(i+1))
	if i != N_FIESTA_freq-1:
		ax.set_xticks([])
	else:
		plt.xlabel('rotation phase')
plt.savefig('shift_function_time-series.png')
plt.show()

# fig, axes = plt.subplots(figsize=(10, 6))
# for i in range(1):
# 	plt.plot(rotation_phase, power_spectrum[i,:], color=colors[i], marker='o', alpha=alpha, label=r'$k={}$'.format(i+1))
# plt.xlim([0.2,0.8])
# plt.xlabel('Stellar rotation phase')
# plt.ylabel('Amplitude A_k')
# plt.legend()
# plt.yscale('log')
# # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# plt.savefig('FIESTA_II_insight_A_k.png')
# plt.show()



#==============================================================================
# Plots 
#==============================================================================
# N_FIESTA_freq = shift_spectrum.shape[1]
N_FIESTA_freq = 5
# print('Because the number of frequencies calculated is large, only the \
# first {:d} frequencies are presented.'.format(N_FIESTA_freq))

plt.rcParams.update({'font.size': 20})
alpha=0.5
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

fig, axes = plt.subplots(figsize=(12, 10))
for i in range(N_FIESTA_freq):
	ax = plt.subplot(N_FIESTA_freq,1,i+1)
	if i ==0:
		plt.title('FIESTA RV')
	plt.plot(np.arange(N_file)/100, shift_spectrum[:, i])
	plt.errorbar(np.arange(N_file)/100, shift_spectrum[:, i], err_shift_spectrum[:, i], marker='.', ls='', alpha=0.5)
	plt.ylabel(r'mode ${%d}$ [m/s]' %(i+1))
	if i != N_FIESTA_freq-1:
		ax.set_xticks([])
	else:
		plt.xlabel('rotation phase')
plt.savefig('time-series.png')
plt.show()

shift_function = np.zeros(shift_spectrum.shape)
for i in range(shift_spectrum.shape[1]):
	shift_function[:, i] = shift_spectrum[:, i] - RV_gauss

fig, axes = plt.subplots(figsize=(12, 10))
for i in range(N_FIESTA_freq):
	ax = plt.subplot(N_FIESTA_freq,1,i+1)
	if i ==0:
		plt.title('FIESTA RV – OBS RV')
	plt.plot(np.arange(N_file)/100, shift_function[:, i])
	plt.errorbar(np.arange(N_file)/100, shift_function[:, i], err_shift_spectrum[:, i], marker='.', ls='', alpha=0.5)
	plt.ylabel(r'mode ${%d}$ [m/s]' %(i+1))
	if i != N_FIESTA_freq-1:
		ax.set_xticks([])
	else:
		plt.xlabel('rotation phase')
plt.savefig('shift_function_time-series.png')
plt.show()


fig, axes = plt.subplots(figsize=(12, 10))
for i in range(N_FIESTA_freq):
	ax = plt.subplot(N_FIESTA_freq,1,i+1)
	if i ==0:
		plt.title('Amplitudes $A_k$')
	plt.plot(np.arange(N_file)/100, power_spectrum[:, i])
	plt.errorbar(np.arange(N_file)/100, power_spectrum[:, i], err_power_spectrum[:, i], marker='.', ls='', alpha=0.5)
	plt.ylabel(r'mode ${%d}$' %(i+1))
	if i != N_FIESTA_freq-1:
		ax.set_xticks([])
	else:
		plt.xlabel('rotation phase')
plt.savefig('power_spectrum_time-series.png')
plt.show()



plt.plot(RV_gauss, rv_weighted, '.', label='rv_weighted')
plt.xlabel('RV_gauss [m/s]')
plt.ylabel('rv_weighted [m/s]')
plt.legend()
plt.savefig('correlation.png')
plt.show()



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
