

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


# Parametric sigmoid function for fitting
def sigmoid(x, x0, k, m, n): 
	y = m / (1 + np.exp(k*(x-x0))) + n
	return y

# Parametric analytic second derivative of sigmoid 
def seconddevsigmoid(x, x0, k, l, m): 
	y = ( k**2 * l * np.exp(k*(x+x0)) * ( np.exp(k*x)-np.exp(k*x0) )  )   /   ( np.exp(k*x0) + np.exp(k*x) )**3    
	return y  

def gen_graph(adj,cutoff = 0.00001):
	# if cutoff is specified: it is used to build the weighted
	# network with values from cutoff and beyond
	n = len(adj[0])
	G=nx.Graph()
	G.add_nodes_from(range(n))

	for i in range(n):
		for j in range(n):
			if adj[i][j] >= cutoff:
				G.add_edge(i,j, weight = adj[i][j])
	return G


# 	reqires network at equilibrium
def distance_persistence_matrix(
	PDB,DCD,
	dt = 0.02, dist_cutoff = 5.0, start = 0,sampling_freq = 1,
	data_path = "dummy/data/"):
	'''
	assume this is performed on a trajectory at equilibrium
	'''

	ref = mda.Universe(PDB)
	trj = mda.Universe(PDB,DCD)
	L = len(trj.trajectory)

	nframes = trj.trajectory.n_frames
	time = np.arange(0, dt*nframes, dt)
	n_alphas = ref.select_atoms("protein and name CA").n_atoms
	time_range = range(start,L,sampling_freq)

	print "Generating persistence matrix for distance based interactions"
	M_persistence = np.zeros(shape =(n_alphas,n_alphas))

	for t in time_range:
		trj.trajectory[t]
		M = np.zeros(shape =(n_alphas,n_alphas))

		calphas_t = trj.select_atoms("protein and name CA")
		for i in range(n_alphas):
			for j in range(n_alphas):
				norm = np.linalg.norm(calphas_t.atoms[i].position - calphas_t.atoms[j].position)
				if norm <= dist_cutoff:
					M[i][j] = 1
		M_persistence = M_persistence + M

	M_persistence = M_persistence/float(len(time_range))
	# save data
	cut = str(int(dist_cutoff))
	np.savetxt(data_path+'M_eq_persistence_dist_{}.csv'.format(cut), M_persistence)



def build_DCCM_matrix(PDB,DCD,dt = 0.02,
							 data_path = "dummy/",
							 images_path = "dummy/"):

	ref = mda.Universe(PDB)
	trj = mda.Universe(PDB,DCD)
	L = int(len(trj.trajectory))

	nframes = trj.trajectory.n_frames
	time = np.arange(0, dt*nframes, dt)

	ref_frame = ref.select_atoms("name CA").positions.copy() #positions
	n_alphas = len(ref_frame)

	#
	# Dynamic Cross Correlation Matrix between C-alphas
	#

	C_mat = np.zeros(shape= (n_alphas,n_alphas))
	positions = np.zeros(shape = (n_alphas,int(L),3))
	delta = np.zeros(shape = (n_alphas,int(L),3))

	for t in range(0,int(L)):
		trj.trajectory[t]
		align.alignto(trj, ref, select="name CA", weights="mass")
		frame_t = trj.select_atoms("name CA").positions.copy()

		for i in range(n_alphas): #still a position vector of displacement
			positions[i][t] = frame_t[i]

	avg_positions = [np.mean(positions[i],axis = 0) for i in range(n_alphas)]

	# naming them self similiarities but actually is just the ensemble average of th
	# dot product of delta_i with itself (similarity name from the dot product operation)
	self_similarities = []
	for i in range(n_alphas):
		sim_ii = []
		for t in range(0,L):
			delta[i][t] = np.array(positions[i][t] - avg_positions[i])
			sim_ii.append(np.dot(delta[i][t],delta[i][t]))
		self_similarities.append( np.mean(np.array(sim_ii)))

	for i in range(n_alphas):
		for j in range(n_alphas):
			if j != i:
				# performing ensemble average of dot product between displacements
				# of atom i and atom j
				similarity_ij = [np.dot(delta[i][t],delta[j][t]) for t in range(L)]
				similarity_ij = np.mean(np.array(similarity_ij))

				C_mat[i][j] = similarity_ij/np.sqrt(self_similarities[i]*self_similarities[j])

	np.savetxt(data_path+"DCCM_equilibrium_Calpha.csv",C_mat)

	plt.imshow(C_mat, cmap = "bone",extent = [0,201,201,0])
	plt.ylabel("C Alpha")
	plt.xlabel("C Alpha")
	plt.colorbar()
	plt.title("Dynamic Cross Correlation Matrix")
	plt.savefig(images_path+"DCCM_Calpha_equilibrium.png",dpi = 450)
	plt.close()






