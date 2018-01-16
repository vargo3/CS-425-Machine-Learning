import numpy as np
import csv
import sys
import math
import operator
import random

def get_matrix( fname ):
	"This finds the mean value."
	R = []
	with open(fname, 'rU') as f:
		reader = csv.reader(f)
		R = list(reader)
	S = []
	for row in R[1:278]:
		S.append(row[1:-9])
	for i in range(len(S)):
		for j in range(len(S[i])):
			if S[i][j] == '':
				S[i][j] = 0
			else:
				S[i][j] = float(S[i][j])
	return S
	
def variance( lis ):
	"find variance of the list given"
	sum = 0.0;
	sumsq = 0.0;
	for num in lis:
		sum += num
		sumsq += num*num
	sum = (sumsq / len(lis)) - (sum*sum)
	return sum
	
def part1():
	"principle component analysis"
	data = np.array(get_matrix("under5mortalityper1000.csv"))
	u, s, vt = np.linalg.svd(data, full_matrices=False)
	v = vt.T
	eigvals = s**2 / np.cumsum(s)[-1]
	pcs = []
	for row in v:
		pcs.append(row[:2])
	reduced_data = np.dot(data, pcs)
	np.set_printoptions(threshold=np.inf)
	np.savetxt(sys.stdout, reduced_data)
	return

def has_converged( centroids, old_centroids ):
	return (set([tuple(a) for a in centroids]) == set([tuple(a) for a in old_centroids]))

def find_clusters( data, centroids ):
	"Assign all points in data to clusters"
	clusters  = {}
	for x in data:
		bestcentroidkey = min([(i[0], np.linalg.norm(x-centroids[i[0]])) \
					for i in enumerate(centroids)], key=lambda t:t[1])[0]
		try:
			clusters[bestcentroidkey].append(x)
		except KeyError:
			clusters[bestcentroidkey] = [x]
	return clusters
 
def recenter( centroids, clusters ):
	"Re-evaluate centers"
	newcentroids = []
	keys = sorted(clusters.keys())
	for k in keys:
		newcentroids.append(np.mean(clusters[k], axis = 0))
	return newcentroids
 
def find_centers(data, k):
	iteration = 0
	#get initial k random cluster centroids
	old_centroids = random.sample(data, k)
	centroids = old_centroids
	
	#find better centroids
	clusters = {}
	while iteration == 0 or not has_converged(centroids, old_centroids):
		old_centroids = centroids
		clusters = find_clusters(data, centroids)
		centroids = recenter(old_centroids, clusters)
		iteration += 1
	print iteration
	return clusters
	
def max_intra_dist( clusters ):
	max = -1.0
	for clust in clusters.items():
		for pointA in clust[1]:
			for pointB in clust[1]:
				dist = np.linalg.norm(pointA-pointB)
				if max == -1 or dist > max:
					max = dist
	return max

def min_inter_dist( clusters ):
	min = -1.0
	for clustA in clusters.items():
		for pointA in clustA[1]:
			for clustB in clusters.items():
				if clustA[0] != clustB[0]:
					for pointB in clustB[1]:
						dist = np.linalg.norm(pointA-pointB)
						if min == -1 or dist < min:
							min = dist
	return min

def part2_full():
	data = np.array(get_matrix("under5mortalityper1000.csv"))
	u, s, vt = np.linalg.svd(data, full_matrices=False)
	v = vt.T
	pcs = []
	for row in v:
		pcs.append(row[:2])
	clusters = find_centers(data, 5)
	for elem in clusters.items():
		np.savetxt(sys.stdout, np.dot(elem[1], pcs))
		print ''
	max = max_intra_dist(clusters)
	min = min_inter_dist(clusters)
	print max
	print min
	print max/min
	return

def part2_pcs():
	data = np.array(get_matrix("under5mortalityper1000.csv"))
	u, s, vt = np.linalg.svd(data, full_matrices=False)
	v = vt.T
	pcs = []
	for row in v:
		pcs.append(row[:5])
	reduced_data = np.dot(data, pcs)
	clusters = find_centers(reduced_data, 5)
	first_pcs = []
	for row in v:
		first_pcs.append(row[:2])
	for elem in clusters.items():
		#shape_diff = np.array((0, np.array(pcs).shape[0]-5))
		#padded = np.lib.pad(np.array(elem[1]), ((0,shape_diff[0]),(0,shape_diff[1])),'constant', constant_values=(0))
		np.savetxt(sys.stdout, np.dot(elem[1], first_pcs[:5]))
		print ''
	max = max_intra_dist(clusters)
	min = min_inter_dist(clusters)
	print max
	print min
	print max/min
	return