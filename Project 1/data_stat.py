import sys
import math
import operator

def mean( fname, field ):
	"This finds the mean value."
	sum = 0
	num = 0
	with open(fname, 'r') as f:
		for line in f:
			words = line.split()
			if words[3] != "?":
				sum += float(words[field])
				num += 1
	print "mean: ", float(sum) / float(num)
	return

def min( fname, field ):
	"This finds the mean value."
	min = float(1000000)
	with open(fname, 'r') as f:
		for line in f:
			words = line.split()
			if words[3] != "?":
				if min > float(words[field]):
					min = float(words[field])
	print "min: ", min
	return

def max( fname, field ):
	"This finds the mean value."
	max = float(0)
	with open(fname, 'r') as f:
		for line in f:
			words = line.split()
			if words[3] != "?":
				if max < float(words[field]):
					max = float(words[field])
	print "max: ", max
	return

def num_val( fname, field ):
	"This finds the mean value."
	num = 0
	with open(fname, 'r') as f:
		for line in f:
			words = line.split()
			if words[3] != "?":
				num += 1
	print "num: ", num
	return

def SD( fname, field ):
	"This finds the mean value."
	num = 0
	variance = 0;
	with open(fname, 'r') as f:
		for line in f:
			words = line.split()
			if words[3] != "?":
				num += 1
				variance += float(words[field]) * float(words[field])
			variance = variance / num
	print "SD: ", math.sqrt(variance) 
	return

def quartiles( fname, field ):
	"This finds the mean value."
	num = 0
	l = list()
	with open(fname, 'r') as f:
		for line in f:
			words = line.split()
			if words[3] != "?":
				num += 1
				l.append( float(words[field]) )
		l.sort()
		print "quartile 1: ", l[int(num / 4.0)]
		print "quartile 2: ", l[int(num * 2.0 / 4.0)]
		print "quartile 3: ", l[int(num * 3.0 / 4.0)]
	return
