import sys
import numpy as np 
import pandas as pd
import re
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import networkx as nx
import json

'''
Usage: 

python cluster.py <directory to the train/test of any fold of the 5-fold dataset> 
                  <percentage of data for subsampling in decimal> 
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

def subsample(samples, p=0.2):
	return np.random.choice(samples, size=int(p * len(samples)))

def get_features(gs):
	return [x.replace('graph', 'feature') for x in gs]


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
	raise NotImplementedError

def plot_pca(X, y):
	plt.clf()
	plt.scatter(X[:, 0].T, X[:, 1].T, c=y)
	plt.show()
	plt.scatter(X[:, 0].T, X[:, 2].T, c=y)
	plt.show()
	plt.scatter(X[:, 1].T, X[:, 2].T, c=y)
	plt.show()

def k_means(X, n_clusters = 8):
	clu = KMeans(n_clusters=n_clusters)
	clu.fit(X)
	res = clu.labels_
	cen = clu.cluster_centers_
	return res, cen, clu

def euclidian_distance(x1, y1, x2, y2):
	return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def manhattan_distance(x1, y1, x2, y2):
	return np.abs(x1 - x2) + np.abs(y1 - y2)

def prepare_graphs(l, method='cutoff', measurement='average', distance=euclidian_distance, threshold=10, threshold2=30):
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



def write_graph(graph_fname, feature_fname, pca_model, kmeans_model,min_cutoff, max_cutoff, patient_to_labels, f):

	graph = prepare_graphs([graph_fname], threshold=min_cutoff, threshold2=max_cutoff)[0]
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
	print('retriving images in {}'.format(argv[1]))
	gs = get_graphs(argv[1])
	print('retrived {} images'.format(len(gs)))
	print('subsampling with p = {}'.format(argv[2]))
	gss = subsample(gs, p=float(argv[2]))

	# Parse other parameters
	min_cutoff = float(argv[3])
	max_cutoff = float(argv[4])
	k_kmeans = int(argv[5])
	path_json = argv[6]
	path_output = argv[7]
	
	fs = get_features(gs)
	fss = get_features(gss)
	print('reading feature files...')
	Xs, ys = prepare_features(fss)
	print('dimensional reducing via PCA')
	Xs_reduced, pca_obj = pca(Xs)
	print('clustering via K-means')
	labels, centeroids, kmeans_obj = k_means(Xs_reduced, n_clusters=k_kmeans)

	if os.path.isfile(path_output):
		os.remove(path_output)
	f = open(path_output, 'w')
	f.write('%d\n' % len(fs))
	ctr = 0

	patient_to_labels = json.load(open(path_json))

	for i in range(len(fs)):
		write_graph(gs[i], fs[i], pca_obj, kmeans_obj, min_cutoff, max_cutoff, patient_to_labels, f)
		ctr += 1
		if ctr % 100 == 0:
			print('processing %d / %d' % (ctr, len(fs)))
	f.close()

if __name__ == '__main__':
	main(sys.argv)