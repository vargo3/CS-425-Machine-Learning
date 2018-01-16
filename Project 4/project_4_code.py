#Jacob Vargo
#project3.py

import csv
import sys
import random
import numpy as np

def read_csv( fname ):
	f = []
	with open(fname, 'rU') as f:
		reader = csv.reader(f)
		f = list(reader)
	lis = []
	for row in f:
		lis.append([float(i) for i in row])
	return lis

def write_csv( fname, lis ):
	with open(fname, 'wb') as f:
		writer = csv.writer(f)
		writer.writerows(lis)
	return

def split_data( original ):
	o_len = len(original)
	random.shuffle(original)
	training = original[ :o_len * 4/8]
	validation = original[o_len * 4/8: o_len * 5/8]
	test = original[o_len * 5/8: ]
	write_csv("train.txt", training)
	write_csv("valid.txt", validation)
	write_csv("test.txt", test)
	return

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

def get_confusion(predict, true, threshold=0.5):
	confusion = {"TN" : 0, "FN" : 0, "FP" : 0, "TP" : 0}
	for i in range(len(predict)):
		if predict[i] > threshold and true[i] > threshold:
			confusion["TP"] += 1
		elif predict[i] > threshold and true[i] <= threshold:
			confusion["FP"] += 1
		elif predict[i] <= threshold and true[i] > threshold:
			confusion["FN"] += 1
		elif predict[i] <= threshold and true[i] <= threshold:
			confusion["TN"] += 1
		else:
			print 'confusion error'
	return confusion
	
#Sigmoid Function
def sigmoid (x):
	return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
	return x * (1 - x)
	
def part1(epoch, learnRate, hiddenlayer_neurons):
	"atificial Neural Network alorithm"
	original = read_csv("spambase.data")
	split_data(original)
	train = np.array(read_csv("train.txt"))
	valid = np.array(read_csv("valid.txt"))
	test = np.array(read_csv("test.txt"))

	#initialize
	#epoch = 10 #Setting training iterations
	#learnRate=0.01 #Setting learning rate
	#hiddenlayer_neurons = 40 #number of hidden layers neurons
	inputlayer_neurons = train.shape[1]-1 #number of features in data set
	output_neurons = 1 #number of neurons at output layer
	wh=np.random.normal(scale=0.5, size=(inputlayer_neurons,hiddenlayer_neurons))
	bh=np.random.normal(scale=0.5, size=(1,hiddenlayer_neurons))
	wout=np.random.normal(scale=0.5, size=(hiddenlayer_neurons,output_neurons))
	bout=np.random.normal(scale=0.5, size=(1,output_neurons))
	
	#training
	for i in range(epoch):
		if (i+1) % (epoch/5) == 0:
			print 'epoch', i+1
			
		for row in train:
			correct = row[-1]
			row = row[:-1]
			#Forward Propogation
			hidden_layer_input1 = np.dot(row,wh)
			hidden_layer_input = hidden_layer_input1 + bh
			hiddenlayer_activations = sigmoid(hidden_layer_input)
			output_layer_input1 = np.dot(hiddenlayer_activations,wout)
			output_layer_input = output_layer_input1 + bout
			output = sigmoid(output_layer_input)

			#Backpropagation
			E = correct-output
			slope_output_layer = derivatives_sigmoid(output)
			slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
			d_output = E * slope_output_layer
			Error_at_hidden_layer = d_output.dot(wout.T)
			d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
			wout += hiddenlayer_activations.T.dot(d_output) * learnRate
			bout += np.sum(d_output, axis=0,keepdims=True) * learnRate
			wh += np.array([row]).T.dot(d_hiddenlayer) * learnRate
			bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) * learnRate
	
	confusion = {"TN" : 0, "FN" : 0, "FP" : 0, "TP" : 0}
	predict = []
	true = []
	for row in valid:
		correct = row[-1]
		row = row[:-1]
		#Forward Propogation
		hidden_layer_input1=np.dot(row,wh)
		hidden_layer_input=hidden_layer_input1 + bh
		hiddenlayer_activations = sigmoid(hidden_layer_input)
		output_layer_input1=np.dot(hiddenlayer_activations,wout)
		output_layer_input= output_layer_input1 + bout
		output = sigmoid(output_layer_input)[0][0]
		predict.append(output)
		true.append(correct)
	confusion = get_confusion(predict, true)
	print_performance(confusion)
	return



