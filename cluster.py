import sys
import numpy as np 
import pandas as pd
import re
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation as LDA
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import Delaunay
import json
import io # for python3

'''
Usage: 

python cluster.py <directory to the training set of any fold of the 5-fold dataset> 
                  <directory to the validation set of any fold of the 5-fold dataset> 
                  <directory to the testing set of any fold of the 5-fold dataset> 
                  <total number of images to subsample> 
                  <Min distance cutoff for graph construction>
                  <Max distance cutoff for graph construction>
                  <K of the K-means clustering algorithm>
                  <path to the "patient_to_labels.json" file> 
                  <path to the output file>
'''

def get_graphs(path):
	gs = []
	for root, dirs, files in os.walk(path, topdown=False):
		for file in files:
			if 'graph' in file:
				gs.append(os.path.join(root, file))
	return gs

def get_features(gs):
	return [x.replace('graph', 'feature') for x in gs]

def subsample(samples, p=0.2):
	return np.random.choice(samples, size=int(p * len(samples)))

def num_cells(fname):
	return pd.read_csv(gs[i]).drop(columns=['Unnamed: 0']).values.shape[0]

def sample_cells(fname, num):
	res = []
	mat = pd.read_csv(gs[i]).drop(columns=['Unnamed: 0']).values
	rand = np.random.choice(num)
	for i in rand:
		res.append(mat[i])
	return np.array(res)

def subsample_even(samples, num_imgs, patient_to_labels):
	tree = []

	# magic number
	for i in range(8):
		tree.append(dict())
	num_imgs_per_class = int(num_imgs / 8)
	for i in samples:
		patient = i.split('/')[-2]
		class_id = patient_to_labels[i.split('/')[-2]]
		if patient not in tree[class_id]:
			tree[class_id][patient] = []
		tree[class_id][patient].append(i)
	res = []
	for i in tree:
		num_imgs_per_patient = int(num_imgs_per_class / len(i.keys()))
		rest_imgs = num_imgs_per_class - int(num_imgs_per_patient * len(i.keys()))
		tmp = []
		for key in i:
			l = i[key]
			tmp += l
			if len(l) >= num_imgs_per_patient:
				res += list(np.random.choice(l, size=num_imgs_per_patient))
			else:
				for counter in range(int(num_imgs_per_patient / len(l))):
					res += l
				res += list(np.random.choice(l, size=int(num_imgs_per_patient % len(l))))
		res += list(np.random.choice(tmp, size=rest_imgs))
	return res

def prepare_features(l, patient_to_labels = None, label_method='patient'):
	res = []
	y = []
	for idx, i in enumerate(l):
		tmp = pd.read_csv(i).drop(columns=['Unnamed: 0']).values
		if label_method == 'patient':
			y += [idx] * len(tmp)
		elif label_method == 'labels':
			y += [patient_to_labels[i.split('/')[-2]]] * len(tmp)
		if res == []:
			res = tmp
		else:
			res = np.vstack((res, tmp))
	return res , y

def pca(X, n_components = 4):
	p = PCA(n_components=n_components)
	return p.fit_transform(X), p

def lda(X, n_components = 4):
	p = LDA(n_components = n_components)
	return p.fit_transform(X), p

def plot_projected(X, y, model):
	trans = model.fit_transform(X)
	plt.clf()
	plt.scatter(trans[:, 0].T, trans[:, 1].T, c=y)
	plt.show()
	plt.scatter(trans[:, 0].T, trans[:, 2].T, c=y)
	plt.show()
	plt.scatter(trans[:, 1].T, trans[:, 2].T, c=y)
	plt.show()

def plot_pca_elbow_plot(X, n_min, n_max, title, fname):
	x, y = [], []
	for i in range(n_min, n_max):
		_, p = pca(X, n_components = i)
		print('PCA n_components = %d, Variance = %f' % (i, p.explained_variance_))
		x.append(i)
		y.append(p.explained_variance_)
	plt.clf()
	plt.title(title)
	print(x)
	print(y)
	plt.plot(x, y)
	plt.savefig(fname, dpi=300)

def k_means(X, n_clusters = 8):
	clu = KMeans(n_clusters=n_clusters)
	clu.fit(X)
	res = clu.labels_
	cen = clu.cluster_centers_
	return res, cen, clu

def sihouette_coef(X, n_min, n_max, title, fname):
	x, y = [], []
	for i in range(n_min, n_max):
		p = KMeans(n_clusters = i)
		lbl = p.fit_transform(X)
		x.append(i)
		y.append(silhouette_score(X, lbl))
	plt.clf()
	plt.title(title)
	plt.plot(x, y)
	plt.savefig(fname, dpi=300)

def euclidian_distance(x1, y1, x2, y2):
	return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def manhattan_distance(x1, y1, x2, y2):
	return np.abs(x1 - x2) + np.abs(y1 - y2)

def cutoff_graphs(l, method='cutoff', measurement='average', distance=euclidian_distance, threshold=10, threshold2=30):
	res = []
	if method == 'cutoff':
		for i in l:
			csv = pd.read_csv(i).drop(columns=['Unnamed: 0']).values
			if measurement == 'average':
				x = np.mean(np.vstack((csv[:, 0], csv[:, 2])), axis=0)
				y = np.mean(np.vstack((csv[:, 1], csv[:, 3])), axis=0)
			elif measurement == 'cell':
				x = csv[:, 0]
				y = csv[:, 1]
			elif measurement == 'nuc':
				x = csv[:, 2]
				y = csv[:, 3]
			G = nx.Graph()
			G.add_nodes_from(range(len(x)))
			for source in range(len(x)):
				for target in range(len(x)):
					if distance(x[source], y[source], x[target], y[target]) > threshold and distance(x[source], y[source], x[target], y[target]) < threshold2 :
						G.add_edge(source, target)
			res.append(G)
	return res

def triangle_graph(l, measurement = 'average'):
	res = []
	for i in l:
		csv = pd.read_csv(i).drop(columns=['Unnamed: 0']).values
		if measurement == 'average':
			x = np.mean(np.vstack((csv[:, 0], csv[:, 2])), axis=0)
			y = np.mean(np.vstack((csv[:, 1], csv[:, 3])), axis=0)
		elif measurement == 'cell':
			x = csv[:, 0]
			y = csv[:, 1]
		elif measurement == 'nuc':
			x = csv[:, 2]
			y = csv[:, 3]
		points = np.zeros((len(csv), 2))
		points[:, 0] = x 
		points[:, 1] = y
		tri = Delaunay(points)
		G = nx.Graph()
		for triangle in tri.simplices:
			G.add_edge(triangle[0], triangle[1])
			G.add_edge(triangle[0], triangle[2])
			G.add_edge(triangle[1], triangle[0])
			G.add_edge(triangle[1], triangle[2])
			G.add_edge(triangle[2], triangle[0])
			G.add_edge(triangle[2], triangle[1])
		res.append(G)
	return res


def write_graph(graph_fname, feature_fname, pca_model, kmeans_model,min_cutoff, max_cutoff, patient_to_labels, f):

	graph = triangle_graph([graph_fname])[0]
	feature, y = prepare_features([feature_fname], patient_to_labels = patient_to_labels, label_method = 'labels')
	graph_label = y[0]
	node_labels = kmeans_model.predict(pca_model.transform(feature))

	# print number of nodes and graph label
	f.write('%d %d\n' % (len(graph), graph_label))

	# print the graph
	for node in graph.nodes:
		targets = graph.adj[node]
		num_edges = len(targets)
		node_label = node_labels[node]
		edges_str = ' '.join([str(x) for x in targets])
		f.write('%d %d %s\n' % (node_label, num_edges, edges_str))


def main(argv):

	# Parse other parameters
	min_cutoff = float(argv[5])
	max_cutoff = float(argv[6])
	k_kmeans = int(argv[7])
	path_json = argv[8]
	path_output = argv[9]
	valid_path_output = argv[10]
	test_path_output = argv[11]

	patient_to_labels = json.load(open(path_json))

	print('retriving images in training set {}'.format(argv[1]))
	gs = get_graphs(argv[1])
	print('retriving images in validation set {}'.format(argv[2]))
	valid_gs = get_graphs(argv[2])
	valid_fs = get_features(valid_gs)
	print('retriving images in testing set {}'.format(argv[3]))
	test_gs = get_graphs(argv[3])
	test_fs = get_features(test_gs)

	print('retrived {} images'.format(len(gs)))
	print('subsampling with total number of imgs = {}'.format(argv[4]))
	gss = subsample_even(gs, int(argv[4]), patient_to_labels)
	
	fs = get_features(gs)
	fss = get_features(gss)
	print('reading feature files...')
	Xs, ys = prepare_features(fss)
	print('getting elbow plot of PCA/LDA...')
	plot_pca_elbow_plot(Xs, 1, 10, 'Elbow plot of PCA from 1 to 20 components', 'pca_elbow.png')
	print('dimensional reducing via PCA')
	Xs_reduced, pca_obj = pca(Xs)
	print('getting sihouette coefficient for K-means...')
	sihouette_coef(Xs, 5, 20, 'Elbow plot of silhouette coefficient from 1 to 20 components', 'kmeans_elbow.png')
	print('clustering via K-means')
	labels, centeroids, kmeans_obj = k_means(Xs_reduced, n_clusters=k_kmeans)

	if os.path.isfile(path_output):
		os.remove(path_output)
	ctr = 0

	graph_contents_stream = io.StringIO()

	for i in range(len(fs)):
		if pd.read_csv(gs[i]).drop(columns=['Unnamed: 0']).values.shape[0] >= 4:
			write_graph(gs[i], fs[i], pca_obj, kmeans_obj, min_cutoff, max_cutoff, patient_to_labels, graph_contents_stream)
			ctr += 1
		if ctr % 100 == 0:
			print('processing %d / %d' % (ctr, len(fs)))
	f = open(path_output, 'w')
	f.write('%d\n' % ctr)
	f.write('%s' % graph_contents_stream.getvalue())
	f.close()

	'''
	validation
	'''
	ctr = 0

	graph_contents_stream = io.StringIO()

	for i in range(len(valid_fs)):
		if pd.read_csv(valid_gs[i]).drop(columns=['Unnamed: 0']).values.shape[0] >= 4:
			write_graph(valid_gs[i], valid_fs[i], pca_obj, kmeans_obj, min_cutoff, max_cutoff, patient_to_labels, graph_contents_stream)
			ctr += 1
		if ctr % 100 == 0:
			print('processing %d / %d' % (ctr, len(valid_fs)))
	f = open(valid_path_output, 'w')
	f.write('%d\n' % ctr)
	f.write('%s' % graph_contents_stream.getvalue())
	f.close()

	'''
	testing
	'''
	ctr = 0

	graph_contents_stream = io.StringIO()

	for i in range(len(test_fs)):
		if pd.read_csv(test_gs[i]).drop(columns=['Unnamed: 0']).values.shape[0] >= 4:
			write_graph(test_gs[i], test_fs[i], pca_obj, kmeans_obj, min_cutoff, max_cutoff, patient_to_labels, graph_contents_stream)
			ctr += 1
		if ctr % 100 == 0:
			print('processing %d / %d' % (ctr, len(test_fs)))
	f = open(test_path_output, 'w')
	f.write('%d\n' % ctr)
	f.write('%s' % graph_contents_stream.getvalue())
	f.close()

if __name__ == '__main__':
	main(sys.argv)