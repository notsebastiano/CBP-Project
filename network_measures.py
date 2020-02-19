'''
auxiliary library to perform the calculation of network measures on adjacency matrices / graphs

author: Sebastiano Bontorin <sebastiano.bontorin@studenti.unitn.it>
'''
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
#from mpl_toolkits.mplot3d import Axes3D

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

def get_degrees(G):
	deg = []
	l = G.number_of_nodes()
	for i in range(l):
		a = G.degree(weight = 'weight')[i]
		deg.append(a)
	return deg

#
# main functions
#

def small_worldness_vs_persistence(adj,upper_threshold,delta,
								   images_path,data_path,name):
	''' 
	These algorithms require a fully connected adjacency matrix.
	Performs measures of clustering coefficient and average
	shortest path length for every graph reconstructed from 
	threshold values. 

	------
	Args:
			- adj: Adjacency matrix - numpy.ndarray
			- upper_threshold: for threshold values
			- delta: increment in threshold_values
			- data_path: path to save C and L data as functions of threhsold_values
			- images_path: path to save the plot
	'''

	threshold_values = np.arange(1e-5,upper_threshold,delta)
	G_0 = gen_graph(adj,cutoff = 1e-5)
	G_2 = gen_graph(adj,cutoff = upper_threshold)

	if nx.is_connected(G_2) is not True:
		raise ValueError("Reconstructed graph for upper_threshold chosen is not connected")

	avg_c0 = nx.algorithms.cluster.average_clustering(G_0,weight='weight')
	avg_L0 = nx.average_shortest_path_length(G_2, weight='weight')
	n = len(adj[0])

	C,L = [],[]
	for val in threshold_values:
		print "threshold value reached:",val
		G = gen_graph(adj,cutoff = val)
		avg_c = nx.algorithms.cluster.average_clustering(G,weight='weight')
		avg_L = nx.average_shortest_path_length(G, weight='weight')

		C.append(avg_c/float(avg_c0))
		L.append(avg_L/float(avg_L0))

		# saving data.
	storing_data = np.array([threshold_values,C,L])
	np.savetxt(data_path + "C_vs_L_{}.csv".format(name), storing_data)
	# generate plot
	fig, ax1 = plt.subplots()

	color = 'tab:red'
	ax1.set_xlabel('Persistence Threshold')
	ax1.set_ylabel('Clustering', color=color)
	ax1.plot(threshold_values, C, color=color)
	ax1.tick_params(axis='y', labelcolor=color)
	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	color = 'tab:blue'
	ax2.set_ylabel('Avg Path Length', color=color)  # we already handled the x-label with ax1
	ax2.plot(threshold_values, L, color=color)
	ax2.tick_params(axis='y', labelcolor=color)

	plt.title('Clustering Coefficient and Average Path Length')
	plt.savefig(images_path+"C_vs_L_{}.png".format(name),dpi=450)
	plt.close()


def sizeGCC_vs_persistence(adj,threshold_values,
						   name,images_path,data_path,
						   sigmoid_guess = 0.4,
						   show = True):
	'''
	Identify largest connected component as function of threshold for a set
	of possible threshold_values. For the series of GCC size then fits a sigmoid 
	and identifies critical threshold using sigmoid flex
	
	! WORKING WITH THIS CHOICE OF THRESHOLD VALUES !
	Different values require to change the way data is padded (different delta):

	> threshold_values = np.arange(0.0,0.999,0.01)

	------
	Args:
			- adj: Adjacency matrix - numpy.ndarray
			- threshold_values = np.arange(0.0,0.999,0.01)
			- name: name for the file
			- data_path: path to save C and L data as functions of threhsold_values
			- images_path: path to save the plot

	Kwargs:
			- sigmoid_guess: first guess for the sigmoid flex fit, consider
							 using show=True to cjeck visually if fitting is performed
							 properly
			- show: show fitted GCC size
	'''
	n = len(adj[0])
	G_size = []
	for val in threshold_values:
		print "threshold value reached:",val
		G = gen_graph(adj,cutoff = val)
		Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
		G_size.append(len(Gcc[0])/float(n))

	# saving data.
	storing_data = np.array([threshold_values,G_size])
	np.savetxt(data_path + "GCCsize_persistence_{}.csv".format(name), storing_data)

	# fitting a sigmoid - padding
	y = np.array(G_size)
	y_before,y_after = np.ones(50),np.ones(50)*min(G_size)
	y_ext = np.concatenate((y_before,y,y_after))
	x_ext = np.arange(-0.50, 1.50, 0.01)

	# fit sigmoid on padded data
	popt, pcov = curve_fit(sigmoid, x_ext, y_ext, p0=(2,2,10,10), maxfev=10000)
	xplot = np.linspace(max(x_ext),min(x_ext))
	yplot = sigmoid(xplot, *popt)

	flex = fsolve(seconddevsigmoid, sigmoid_guess, args=tuple(popt), maxfev=20000)
	print "flex found for {} is :".format(name),flex

	x = threshold_values
	xplot_reduced = np.linspace(max(x),min(x))
	plt.plot(threshold_values,G_size,'-o',label = 'GCC Size')
	plt.plot(xplot_reduced, sigmoid(xplot_reduced, *popt), label='Sigmoid Fit')
	plt.plot(flex,sigmoid(flex,*popt),'o',label = 'Critical value', color = 'red')

	plt.xlabel('Threshold value')
	plt.ylabel('Size of GCC (normalized)')
	plt.legend(loc='best')
	plt.title('Size of GCC vs Threshold values')
	plt.savefig(images_path+"GCCsize_persistence_{}.png".format(name),dpi=450)
	if show == True:
		plt.show()
	plt.close()






