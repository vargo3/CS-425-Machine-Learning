#Jacob Vargo

import csv
import sys
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import svm

def read_csv( fname ):
	f = []
	with open(fname, 'rU') as f:
		reader = csv.reader(f)
		f = list(reader)
	lis = []
	for row in f:
		if len(f) <= len(lis):
			print 'length error:', len(f), len(lis)
		tmp = [float(i) for i in row[:-1]]
		tmp.append(float(row[-1]))
		lis.append(tmp)
	return lis

def read_csv_first( fname ):
	f = []
	with open(fname, 'rU') as f:
		reader = csv.reader(f, delimiter=' ', skipinitialspace=True)
		f = list(reader)
	lis = []
	for line in f:
		if len(f) < len(lis):
			print 'length error:', len(f), len(lis)
		row = line[3:]
		tmp = [float(i) for i in row[:-1]]
		tmp.append(float(row[-1]))
		if len(tmp) != 11:
			print 'tmp has bad length:', len(tmp)
			print line
			print tmp
			#sys.exit()
		lis.append(tmp)
	return lis

def write_csv( fname, lis ):
	with open(fname, 'wb') as f:
		writer = csv.writer(f)
		writer.writerows(lis)
	return

def split_data( original, prefix='' ):
	o_len = len(original)
	random.shuffle(original)
	training = original[ :o_len * 2/4]
	validation = original[o_len * 2/4: o_len * 3/4]
	test = original[o_len * 3/4: ]
	write_csv(prefix+"train.txt", training)
	write_csv(prefix+"valid.txt", validation)
	write_csv(prefix+"test.txt", test)
	return
	
	
	
def part2():
	original = read_csv_first("vowel-context.data")
	pre = 'p2_'
	#split_data(original, prefix=pre)
	train = np.array(read_csv(pre+"train.txt"))
	valid = np.array(read_csv(pre+"valid.txt"))
	test = np.array(read_csv(pre+"test.txt"))
	X = []
	y = []
	for row in train:
		X.append(row[:-1])
		y.append(row[-1])
	v_X = []
	v_y = []
	for row in valid:
		v_X.append(row[:-1])
		v_y.append(row[-1])
	t_X = []
	t_y = []
	for row in test:
		t_X.append(row[:-1])
		t_y.append(row[-1])
	scaler = StandardScaler()
	#print X
	scaler.fit(X)
	X = scaler.transform(X)
	v_X = scaler.transform(v_X)
	t_X = scaler.transform(t_X)
	
	#gamma_lis = [.0001, .001, .01, .1, 1, 10]
	#C_lis = [.01, .1, .5, .75, 1, 1.25, 1.5, 2, 5, 10, 20, 30, 40, 50, 100, 200]
	gamma_lis = np.random.uniform(low=.01, high=1.0, size=(100,))
	C_lis = np.random.uniform(low=5, high=20, size=(100,))
	svc = svm.SVC()
	clf = GridSearchCV(svc, {'C': C_lis, 'gamma': gamma_lis}, n_jobs=-1)
	clf.fit(X, y)
	print clf.best_score_
	print clf.best_params_
	clf2 = svm.SVC(C=clf.best_params_['C'], gamma=clf.best_params_['gamma'])
	clf2.fit(X,y)
	predictions = clf2.predict(v_X)
	correct = 0
	for i in range(len(v_y)):
		if v_y[i] == predictions[i]:
			correct += 1
	accuracy = correct / float(len(v_y))
	print accuracy
	return
	best_accuracy = 0
	for gamma in gamma_lis:
		for C in C_lis:
			clf.set_params(C=C, gamma=gamma)
			clf.fit(X, y)
			predictions = clf.predict(v_X)
			correct = 0
			for i in range(len(v_y)):
				if v_y[i] == predictions[i]:
					correct += 1
			accuracy = correct / float(len(v_y))
			#print accuracy
			if accuracy > best_accuracy:
				best_params = (C, gamma, accuracy)
				best_accuracy = accuracy
	print best_params
	return



if __name__ == "__main__":
	part2()



