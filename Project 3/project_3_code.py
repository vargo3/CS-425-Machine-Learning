#Jacob Vargo
#project3.py

import csv
import sys
import math
import operator

class patient:
	"""patient data"""
	id = 0
	data = []
	isBenign = True

def get_matrix( fname ):
	"This finds the mean value."
	f = []
	with open(fname, 'rU') as f:
		reader = csv.reader(f)
		f = list(reader)
	mat = {}
	for row in f:
		pat = patient()
		pat.id = int(row[0])
		pat.data = []
		if row[-1] == "4":
			pat.isBenign = False
		elif row[-1] == "2":
			pat.isBenign = True
		else:
			print "data error:", row[-1]
		for elem in row[1:-2]:
			if elem == '?':
				elem = 1
			pat.data.append(int(elem))
		mat.setdefault(pat.id, pat)
	return mat

def print_performance(confusion):
	TN = confusion["TN"]
	TP = confusion["TP"]
	FN = confusion["FN"]
	FP = confusion["FP"]
	print "TN: ", TN
	print "TP: ", TP
	print "FN: ", FN
	print "FP: ", FP
	if (TN + TP + FN + FP) != 0:
		acc = (TN + TP) * 1.0 / (TN + TP + FN + FP)
	else:
		acc = -1
	print "Accuracy: ", acc
	if (TP + FN) != 0:
		TPR = TP * 1.0 / (TP + FN)
	else:
		TPR = -1
	print "TPR:", TPR
	if (TP + FP) != 0:
		PPV = TP * 1.0 / (TP + FP)
	else:
		PPV = -1
	print "PPV:", PPV
	if (TN + FP) != 0:
		TNR = TN * 1.0 / (TN + FP)
	else:
		TNR = -1
	print "TNR:", TNR
	if (PPV == -1) or (TPR == -1):
		fscore = -1
	else:
		fscore = PPV * TPR * 1.0 / (PPV + TPR)
	print "f score:", fscore

def get_confusion(predict, true):
	confusion = {"TN" : 0, "FN" : 0, "FP" : 0, "TP" : 0}
	for pItem in predict:
		if pItem.id in true:
			tVal = true[pItem.id]
			if pItem.isBenign == True and tVal.isBenign == True:
				confusion["TN"] += 1
			elif pItem.isBenign == True and tVal.isBenign == False:
				confusion["FN"] += 1
			elif pItem.isBenign == False and tVal.isBenign == True:
				confusion["FP"] += 1
			elif pItem.isBenign == False and tVal.isBenign == False:
				confusion["TP"] += 1
	return confusion

def distance(point1, point2):
	#the dimensionality of point1 and point2 must match
	dist = 0
	for x in range(len(point1)):
		dist += pow((point1[x] - point2[x]), 2)
	return math.sqrt(dist)

def get_neighbors(training, point, k):
	similiarities = []
	for x in training.itervalues():
		dist = distance(point.data, x.data)
		similiarities.append((x, dist))
	similiarities.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(similiarities[x][0])
	return neighbors

def neighbor_predict(neighbors):
	vote_true = 0
	vote_false = 0
	for x in range(len(neighbors)):
		if neighbors[x].isBenign:
			vote_true +=1
		else:
			vote_false +=1
	if vote_true > vote_false:
		return True
	else:
		return False
		
def part1(k):
	"k-nearest neighbor alorithm"
	filename = "breast-cancer-wisconsin.data"
	original = get_matrix(filename)
	o_len = len(original)
	training = dict(original.items()[ :o_len * 1/2])
	validation = dict(original.items()[o_len * 1/2: o_len * 3/4])
	test = dict(original.items()[o_len * 3/4: ])
	
	predictions = []
	for x in validation.itervalues():
		neighbors = get_neighbors(training, x, k)
		result = neighbor_predict(neighbors)
		pat = patient()
		pat.id = x.id
		pat.data = x.data
		pat.isBenign = result
		predictions.append(pat)
	confusion = get_confusion(predictions, original)
	print_performance(confusion)
	
	predictions = []
	for x in test.itervalues():
		neighbors = get_neighbors(training, x, k)
		result = neighbor_predict(neighbors)
		pat = patient()
		pat.id = x.id
		pat.data = x.data
		pat.isBenign = result
		predictions.append(pat)
	confusion = get_confusion(predictions, original)
	print_performance(confusion)



def get_prob(index, value, data):
	p = 0
	for row in data:
		if row[index] < value:
			p += 1
	return float(p) / len(data)

def split_attribute(data):
	min_impurity = 999
	for i in range(len(pat.data)):
		for row in range(len(data)):
			p = get_prob(i, 
			impurity = -(p * math.log(p, 2)) - ((1-p) * math.log(1-p, 2)) #entropy
			#impurity = 2 * p * (1-p) #gini
			#impurity = 1 - max(p, 1-p) #misclassification error
			if impurity < min_impurity:
				min_impurity = impurity
				bestf = i
	return bestf

def generate_tree(data, tree, depth, max_depth, theta):
	for attr in range(9):
		if depth == max_depth:
			#create leaf labeled by majority class in data
			vote_true = 0
			for row in data:
				if row.isBenign == True:
					vote_true += 1
			if vote_true > len(data):
				leaf = {True : True}
			else:
				leaf = {False : False}
			return leaf
		i = split_attribute(data)
		sub_data = []
		for row in data:
			if row.data[attr] < i:
				sub_data.append(row)
		tree[True] = generate_tree(sub_data, tree, depth+1, max_depth, theta)
		tree[False] = generate_tree(sub_data, tree, depth+1, max_depth, theta)
	return tree

def decision_predict(point, tree):
	vote_true = 0
	for attr in range(9):
	
	if vote_true > 4:
		return True
	else:
		return False

def part2(data):
	"Decision tree classifier"
	filename = "breast-cancer-wisconsin.data"
	original = get_matrix(filename)
	o_len = len(original)
	training = dict(original.items()[ :o_len * 1/2])
	validation = dict(original.items()[o_len * 1/2: o_len * 3/4])
	test = dict(original.items()[o_len * 3/4: ])
	
	tree = {}
	tree = generate_tree(training, tree)
	
	predictions = []
	for x in validation.itervalues():
		pat = patient()
		pat.id = x.id
		pat.data = x.data
		pat.isBenign = decision_predict(x, tree)
		predictions.append(pat)
	confusion = get_confusion(predictions, original)
	print_performance(confusion)
	
	predictions = []
	for x in test.itervalues():
		pat = patient()
		pat.id = x.id
		pat.data = x.data
		pat.isBenign = decision_predict(x, tree)
		predictions.append(pat)
	confusion = get_confusion(predictions, original)
	print_performance(confusion)


	
