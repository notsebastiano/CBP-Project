'''

code used to reconstruct networks from pyinteraph and DCCM data and perform different measures
using the libraries created

analysis for the wild type and resembles the same structure

'''

import network_measures as measures
import network_creation as creation
import standard_measures as standard

import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from mpl_toolkits.mplot3d import Axes3D

from MDAnalysis.analysis import align
import MDAnalysis.analysis.rms as rms

import networkx as nx
import scipy
import scipy.stats

from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.signal import medfilt



# load trajectories and topologies
if 1:
	print "loading files for A89G mutated recoverin in membrane"
	PDB = "/Users/sebastiano/Desktop/A89G_analysis/recmut_memb_data/firstdcdframe_run8.pdb"
	PSF = "/Users/sebastiano/Desktop/G05-rec-membrane/recmut-membrane_production/run_1/recmut_memb_ionized.psf"
	DCD = "/Users/sebastiano/Desktop/A89G_analysis/recmut_memb_data/equilibrium_run8to41.dcd" 

	ref = mda.Universe(PDB)
	trj = mda.Universe(PDB,DCD)
	L = len(trj.trajectory)
	print("number of frames in dcd: ",L)

	nframes = trj.trajectory.n_frames
	dt = 0.02 # ns per frame 10000 dcd means 20000 fs or 20 ps or 0.02 ns
	time = np.arange(0, dt*nframes, dt)
	n_alphas = ref.select_atoms("protein and name CA").n_atoms

	range_resid = np.arange(n_alphas) + 2

# network generation - from equilibrium traj and pyinteraph data
# remembering tu subselect the protein out of the full net with the membrane

# generate again rmsd
standard.rmsd_heatmap(PDB,DCD,PSF, dt = 0.02, sel = "name CA",
	data_path = "analysis_A89G/data/",
	images_path = "analysis_A89G/images/")


#
# GENERATE DATA: generate dist 5 persistence and DCCM matrices
#
if 0:
	creation.distance_persistence_matrix(
		PDB,DCD,
		dt = 0.02, dist_cutoff = 5.0, start = 0,sampling_freq = 1,
		data_path = "analysis_A89G/data/"
		)

	creation.build_DCCM_matrix(PDB,DCD,dt = 0.02,
		data_path = "analysis_A89G/data/",
		images_path = "analysis_A89G/images/")

	# load matrices generated:
	dist_data = np.loadtxt("analysis_A89G/data/M_eq_persistence_dist_5.csv")
	# import pyint data
	sb_data = np.loadtxt("analysis_A89G/data/sb_interactions.dat")
	hb_data = np.loadtxt("analysis_A89G/data/hb_interactions.dat")
	hc_data = np.loadtxt("analysis_A89G/data/hc_interactions.dat")

	# aggreagate matrices
	# aggregate pyint matrices and dist 5 taking the max value out of persistences for every entry ij

	aggregate_pyint = np.zeros(shape = (n_alphas,n_alphas))
	aggregate_pyint_d5 = np.zeros(shape = (n_alphas,n_alphas))

	for i in range(n_alphas):
		for j in range(n_alphas):
			val = max([sb_data[i][j],hc_data[i][j],hb_data[i][j]])
			val2 = max([sb_data[i][j],hc_data[i][j],hb_data[i][j],dist_data[i][j]])
			aggregate_pyint[i][j] = val
			aggregate_pyint_d5[i][j] = val2

	# save aggregated data
	np.savetxt('analysis_A89G/data/M_eq_pyint_aggregated.csv', aggregate_pyint)
	np.savetxt('analysis_A89G/M_eq_dist5_pyint_aggregated.csv', aggregate_pyint_d5)


# load data
DCCM = np.loadtxt("analysis_A89G/data/DCCM_equilibrium_Calpha.csv")
aggregate_pyint = np.loadtxt("analysis_A89G/data/M_eq_pyint_aggregated.csv")
aggregate_pyint_d5 = np.loadtxt("analysis_A89G/data/M_eq_pyint_dist5_aggregated.csv")

#
# network analysis
#

if 0:
	# small worldness (requires fully connectedness)
	measures.small_worldness_vs_persistence(adj = aggregate_pyint_d5,
											upper_threshold=0.999,
											delta = 0.001,
											images_path="analysis_A89G/images/",
											data_path = "analysis_A89G/data/",
											name = "pyint_d5")


if 0:
	# size gcc
	measures.sizeGCC_vs_persistence(adj = aggregate_pyint,
						   threshold_values = np.arange(0.0,0.999,0.01),
						   name = "aggregated_pyint",
						   images_path = "analysis_A89G/images/",
						   data_path = "analysis_A89G/data/",
						   sigmoid_guess = 0.7,
						   show = True)

#
# DCCM network analysis
#

if 0:
	# size gcc for DCCM
	measures.sizeGCC_vs_persistence(adj = DCCM,
						   threshold_values = np.arange(0.0,0.999,0.01),
						   name = "DCCM",
						   images_path = "analysis_A89G/images/",
						   data_path = "analysis_A89G/data/",
						   sigmoid_guess = 0.7,
						   show = True)
if 0:
	'''
	small worldness (requires fully connectedness) of DCCM, especially at high thresholds.
	Therefore I decided to add diagonal elements where they where missing with the assumption
	that high correlation between residues connected by peptide bonds is likely to happen
	'''

	### add diagonals
	for i in range(n_alphas):
		for j in range(n_alphas):
			if j == i-1 or j == i or j == i + 1:
				if DCCM[i][j] <= 0.90:
					DCCM[i][j] = 0.90

	measures.small_worldness_vs_persistence(adj = DCCM,
		upper_threshold=0.95,
		delta = 0.01,
		images_path="analysis_A89G/images/",
		data_path = "analysis_A89G/data/",
		name = "DCCM_mod")

#
# combined plots of results
#

if 0:
	# plotting together  GCC sizes and small_worldness
	images_path = "analysis_A89G/images/"

	data_CL = np.loadtxt("analysis_A89G/data/C_vs_L_pyint_d5.csv")
	x_CL,C,L = data_CL[0],data_CL[1],data_CL[2]

	data_GCC_pyint = np.loadtxt("analysis_A89G/data/GCCsize_persistence_aggregated_pyint.csv")
	x_GCC_pyint, y_GCC_pyint = data_GCC_pyint[0],data_GCC_pyint[1]


	fig, ax1 = plt.subplots()
	xx = np.arange(0.65,0.8,0.001)
	for x in xx:
		plt.axvline(x=x,color = 'orange',alpha = 0.04)
	color = 'tab:red'
	ax1.set_xlabel('Persistence Threshold')
	ax1.set_ylabel('Clustering', color=color)
	ax1.plot(x_CL, C,'-o', color=color,markersize = 1.5,alpha = 0.7,label = "avg C")
	ax1.tick_params(axis='y', labelcolor=color)

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	color = 'tab:blue'
	ax2.set_ylabel('Avg L - Gcc size Pyint', color='black')  # we already handled the x-label with ax1
	ax2.plot(x_CL, L, '-o',color=color,markersize = 1.5,alpha = 0.7,label = "avg L")
	ax2.tick_params(axis='y', labelcolor=color)

	ax2.plot(x_GCC_pyint, y_GCC_pyint,'-o', color='black',markersize = 2,alpha = 0.7,label ="GCC size pyint")
	plt.legend(loc = 'lower left')

	plt.title('Avg C - Avg L - Gcc size - Pyint network A89G')
	plt.savefig(images_path+"CL_GCC_pyint_combined.png",dpi=450)
	plt.show()
	plt.close()

if 0:
	images_path = "analysis_A89G/images/"
	data_CL = np.loadtxt("analysis_A89G/data/C_vs_L_DCCM_mod.csv")
	x_CL,C,L = data_CL[0],data_CL[1],data_CL[2]

	data_GCC_DCCM = np.loadtxt("analysis_A89G/data/GCCsize_persistence_DCCM.csv")
	x_GCC_DCCM, y_GCC_DCCM = data_GCC_DCCM[0],data_GCC_DCCM[1]


	fig, ax1 = plt.subplots()
	xx = np.arange(0.65,0.8,0.001)
	for x in xx:
		plt.axvline(x=x,color = 'orange',alpha = 0.04)
	color = 'tab:red'
	ax1.set_xlabel('Persistence Threshold')
	ax1.set_ylabel('Clustering', color=color)
	ax1.plot(x_CL, C,'-o', color=color,markersize = 1.5,alpha = 0.7,label = "avg C")
	ax1.tick_params(axis='y', labelcolor=color)

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	color = 'tab:blue'
	ax2.set_ylabel('Avg L - Gcc size DCCM', color='black')  # we already handled the x-label with ax1
	ax2.plot(x_CL, L, '-o',color=color,markersize = 1.5,alpha = 0.7,label = "avg L")
	ax2.tick_params(axis='y', labelcolor=color)

	ax2.plot(x_GCC_DCCM, y_GCC_DCCM,'-o', color='black',markersize = 2,alpha = 0.7,label ="GCC size DCCM")
	plt.legend(loc = 'lower left')

	plt.title('Avg C - Avg L - Gcc size - DCCM network A89G')

	plt.savefig(images_path+"CL_GCC_DCCM_combined.png",dpi=450)
	plt.show()
	plt.close()














