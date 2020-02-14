import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from mpl_toolkits.mplot3d import Axes3D

from MDAnalysis.analysis import align
import MDAnalysis.analysis.rms as rms

import networkx as nx
import scipy
import scipy.stats

import MDAnalysis.analysis.pca as pca
from scipy.signal import medfilt


def gen_graph(adj,cutoff = 0.0):
	n = len(adj[0])
	G=nx.Graph()
	G.add_nodes_from(range(n))

	for i in range(n):
		for j in range(n):
			if adj[i][j] > cutoff:
				G.add_edge(i,j, weight = adj[i][j])
	return G

'''
if 1:
	PDB = "/Users/sebastiano/Desktop/A89G_analysis/recmut_memb_data/first_dcd_frame_run1.pdb"
	#PDB = "/Users/sebastiano/Desktop/A89G_analysis/recmut_memb_data/firstdcdframe_run8.pdb"
	PSF = "/Users/sebastiano/Desktop/G05-rec-membrane/recmut-membrane_production/run_1/recmut_memb_ionized.psf"
	# put full DCD HERE
	DCD = "/Users/sebastiano/Desktop/A89G_analysis/recmut_memb_data/recmut_run1to41.dcd" #full run
	#DCD = "/Users/sebastiano/Desktop/A89G_analysis/recmut_memb_data/equilibrium_run8to41.dcd" 
'''



def rmsd_heatmap(PDB,DCD,PSF, dt = 0.02, sel = "name CA",
	data_path = "analysis_A89G/data/",
	images_path = "analysis_A89G/images/"):
	'''
	- heatmap[s][t] is root mean square distance of atom s at time t wrt ref frame:
	  once heatmap is computed after aligning the traj I can recall rmsd and rmsf by summing over
	  columns and rows respectively
	- this full rmsd of fitted frame is validated through comparison with:
	  rms.rmsd(ref_frame, frame_i, center=True, superposition = True)
	'''
	ref = mda.Universe(PSF,PDB)
	trj = mda.Universe(PSF,DCD)
	L = len(trj.trajectory)
	nframes = trj.trajectory.n_frames
	time = np.arange(0, dt*nframes, dt)

	ref_frame = ref.select_atoms(sel).positions.copy()
	n_alphas = len(ref_frame)

	if 1:
		print "generating heatmap for {} c-alpha in a {} frames trajectory".format(str(n_alphas),str(nframes))
		heatmap = np.zeros(shape =(n_alphas,int(L))) 
		full_rmsd,rmsf = [],[]

		for t in range(0,int(L)):
			trj.trajectory[t]
			align.alignto(trj, ref, select="name CA", weights="mass")
			frame_t = trj.select_atoms("name CA").positions.copy()

			for j in range(0,n_alphas):
				rmsd_j = np.linalg.norm( frame_t[j] - ref_frame[j])
				heatmap[j][t] = rmsd_j

			full_rmsd.append( np.sqrt(np.mean( heatmap[:,t]**2)))

		for i in range(0,n_alphas):
			rmsf.append( np.sqrt( np.mean( heatmap[i]**2)))

		# save data
		np.savetxt(data_path+"heatmap.csv",heatmap)
		np.savetxt(data_path+"rmsd.csv",full_rmsd)
		np.savetxt(data_path+"rmsf.csv",rmsf)

	#full_rmsd = np.loadtxt(data_path+"rmsd.csv")
	#rmsf = np.loadtxt(data_path+"rmsf.csv")

	#
	# generate and save images
	#
	rmsd_filt = medfilt(full_rmsd,kernel_size = 31)
	plt.plot(time,full_rmsd,'teal',alpha = 0.4,label = "RMSD")
	plt.plot(time,rmsd_filt,'teal',label = "Median filter RMSD")
	plt.title("RMSD")
	plt.xlabel("Time [ ns ]")
	plt.ylabel("RMSD [Angstrom]")
	plt.ylim(0, np.max(full_rmsd)+1.0)
	plt.legend(loc='upper left')
	plt.savefig(images_path +"rmsd.png",dpi = 450)

	plt.show()
	plt.close()

	plt.bar(range(1,len(rmsf)+1), rmsf)
	plt.title("Contributions in RMSD")
	plt.xlabel("Residue [C-alpha]")
	plt.ylabel("RMSD [Angstrom]")
	plt.legend(loc='upper left')
	plt.savefig(images_path +"rmsf.png",dpi = 450)
	plt.close()

	if 0:
		# plotting the heatmap
		ext_heatmap = np.zeros(shape= (len(heatmap)*15, int(len(heatmap[0])/2.0)+1))

		for j in range(0,len(heatmap[0]),2):
			for i in range(0,len(heatmap)):
				for k in range(15):
					ext_heatmap[i*15+k][int(j/2.0)] = heatmap[i][j]**2
		plt.imshow(ext_heatmap,cmap = "inferno",extent=[0,200,201,0])
		plt.ylabel("Carbon Alpha")
		plt.xlabel("Time [ns]")
		plt.clim(0,25) # setting upper limit
		plt.title("Atomic deviations heatmap")
		plt.savefig(images_path +"heatmap.png",dpi = 450)
		plt.close()


def spearman_correlation_rmsd(rmsd_path,time_range,images_path):
	# correlation coeff for rmsd
	RMSD = np.loadtxt(rmsd_path)
	pr_corr = []
	for t in time_range:
		pr = scipy.stats.spearmanr(RMSD[t:], RMSD[:len(RMSD)-t])
		pr_corr.append(pr[0])

	plt.plot(time_range,pr_corr)
	plt.title("Spearman Autocorrelation coeff")
	plt.xlabel("Delta T")
	plt.ylabel("spearman r")
	plt.legend(loc='upper left')
	plt.savefig(images_path +"spearmanr_rmsd.png",dpi = 450)
	plt.close()


def compute_measures(PDB,DCD,dt = 0.02, sel = "name CA",
	data_path = "dummy_folder/data",
	images_path = "dummy_folder/images"):
	'''
	computes:
	- RGYR - DISTANCE MEMB PROTEIN - DISTANCE MEMB MYR
	- this function was copied from the mighty fabio.mazza
	'''
	ref = mda.Universe(PDB)
	trj = mda.Universe(PDB,DCD)
	L = len(trj.trajectory)
	nframes = trj.trajectory.n_frames
	time = np.arange(0, dt*nframes, dt)

	### radius of gyration
	protein = trj.select_atoms("backbone")
	Rgyr = np.array([protein.radius_of_gyration() for ts in trj.trajectory])
	np.savetxt(data_path+"rgyr.csv",Rgyr)


	plt.plot(time, Rgyr)
	plt.xlabel("Time (ns)")
	plt.ylabel("RGYR (angstrom)")
	plt.title("Radius Of Gyration")
	plt.savefig(images_path+"rgyr.png",dpi=450)
	plt.close()

	### dist membrane - protein
	# center of mass of the membrane
	memb = trj.select_atoms('resname DGPS DGPC DGPE')
	memb_cdm = np.array([memb.center_of_mass() for ts in trj.trajectory[:]])
	memb_z = memb_cdm[:,2]
	# z of the center of mass of the protein, minus the z of the cdm of the membrane
	prot = trj.select_atoms('protein and name CA')
	prot_cdm = np.array([prot.center_of_mass() for ts in trj.trajectory[:]])
	prot_z = prot_cdm[:,2]

	pm_distance = (prot_z - memb_z) / 10.0
	np.savetxt(data_path+"protein_membrane_dist.csv",pm_distance)

	plt.plot(time, pm_distance)
	plt.xlabel('Time [ns]')
	plt.ylabel('Distance in z-axis [nm]')
	plt.title('Distance between CM Protein and CM Membrane')
	plt.savefig(images_path+"protein_membrane_dist.png",dpi=450)
	plt.close()

	### dist membrane - myr
	# z of the center of mass of the myristoil, minus the z of the cdm of the membrane
	myr = trj.select_atoms('resname GLYM')
	myr_cdm = np.array([myr.center_of_mass() for ts in trj.trajectory[:]])
	myr_z = myr_cdm[:,2]

	mm_distance = (myr_z - memb_z) / 10.0
	mm_distance_smooth = np.convolve(mm_distance, np.ones((20,))/20, mode='valid')
	np.savetxt(data_path+"myr_membrane_dist.csv",mm_distance_smooth)

	plt.plot(time[:len(mm_distance_smooth)], mm_distance_smooth)
	plt.xlabel('Time [ns]')
	plt.ylabel('Distance in z-axis [nm]')
	plt.title('Distance between CM Myristoil and CM Membrane')
	plt.savefig(images_path+"myr_membrane_dist.png",dpi=450)
	plt.close()





'''
PCA
if 0:
	PSF_pca = pca.PCA(trj, select='name CA', align=True) # meglio align o no?
	PSF_pca.run()

	# select components that explain >95 percent of the variance
	n_pcs = np.where(PSF_pca.cumulated_variance > 0.95)[0][0]
	print(np.where(PSF_pca.cumulated_variance > 0.95)[:][0])
	atomgroup = trj.select_atoms('name CA')
	pca_space = PSF_pca.transform(atomgroup, n_components=3)

	## plot of the 3 main components
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	pca3d = ax.scatter(pca_space[:,0], pca_space[:,1], pca_space[:,2], s=2, alpha=0.8, c=np.arange(nframes))
	cbar=fig.colorbar(pca3d)
	cbar.set_label("Frames")

	for angle in range(0, 360):
		ax.view_init(30, angle)
		plt.draw()
		plt.pause(.001)

	plt.title("Plot of first 3 PCA eigenvectors along the traj - A89G")
	plt.show()
'''







