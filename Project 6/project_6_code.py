#Jacob Vargo

import csv
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import rand

reward = 100
matrix_shape = 20 #side length of the 2D square matrix

def main( grid, discount, epsilion, update_factor, trace_decay ):
	V = np.zeros((matrix_shape, matrix_shape)).tolist()
	epochs = 300
	total_episodes = 0
	total_steps = 0
	for i in range(epochs):
		V, num_episodes, num_steps = estimate_V(V, grid, discount, epsilion, update_factor, trace_decay, max_step=600)
		total_episodes += num_episodes
		total_steps += num_steps
	average_reward = 1.0 * reward * epochs / total_episodes 
	average_steps = 1.0 * total_steps / total_episodes
	return V, average_reward, average_steps

def create_world( filename=None, density=0.1, goal=None, do_print=False ):
	grid = []
	if filename != None:
		#assumes that the file is csv and a 20x20 grid with a goal already designated
		#undefined behavior if assumption is false
		grid = read_csv(filename)
	else:
		for i in range(matrix_shape):
			if density != 0:
				row = np.random.randint(0, 1/density, matrix_shape)
			else:
				row = np.random.randint(0, 1, matrix_shape)
			row[row!=1] = 0
			grid.append(row.tolist())
		if goal == None:
			i = random.randint(0, matrix_shape-1)
			j = random.randint(0, matrix_shape-1)
		else:
			i = goal[0]
			j = goal[1]
		grid[i][j] = 2
	if do_print == True:
		for row in grid:
			print row
	return grid

def estimate_V( V, grid, discount, epsilion, update_factor, trace_decay, max_step=50 ):
	total_episodes = 0
	total_steps = 0
	while True:
		total_episodes += 1
		#pick random starting point for robot sufficiently far away from goal
		s_t = (1, 1)
		s_t1 = s_t
		path = {}
		for i in range(max_step):
			total_steps += 1
			s_t1 = pick_next_state(s_t, V, grid, epsilion)
			reward_t1 = 0
			if grid[ s_t1[0] ][ s_t1[1] ] == 2:
				reward_t1 = reward
			#update V(s_t)
			delta = (reward_t1 + (discount * V[ s_t1[0] ][ s_t1[1] ]) - V[ s_t[0] ][ s_t[1] ])
			#V[ s_t[0] ][ s_t[1] ] = V[ s_t[0] ][ s_t[1] ] + (trace_decay * delta)
			path[s_t] = 1
			for s in path:
				V[ s_t[0] ][ s_t[1] ] = V[ s_t[0] ][ s_t[1] ] + (trace_decay * delta * path[s])
				path[s] = discount * update_factor * path[s]
			s_t = s_t1 #move robot
			if grid[ s_t[0] ][ s_t[1] ] == 2:
				#print 'reached goal!'
				return V, total_episodes, total_steps

def pick_next_state( s_t, V, grid, epsilion ):
	#pick a direction to move in greedily based on V
	best = 0
	choices = []
	for i in (-1,1):
		#check if the option is inside the grid and is not going into a wall
		if s_t[0]+i < matrix_shape and 0 <= s_t[0]+i and grid[ s_t[0]+i ][ s_t[1] ] != 1:
			choices.append((s_t[0]+i, s_t[1]))
		if s_t[1]+i < matrix_shape and 0 <= s_t[1]+i and grid[ s_t[0] ][ s_t[1]+i ] != 1:
			choices.append((s_t[0], s_t[1]+i))
	if choices == []:
		print ('choice error! robot cannot move anywhere')
	
	for s in choices:
			if best < V[ s[0] ][ s[1] ]:
				best = V[ s[0] ][ s[1] ]
				s_t1 = s
	#e is to choose randomly sometimes to explore instead of follow best path
	e = np.random.choice([0,1], p=[epsilion, 1-epsilion])
	if best == 0 or e == 0:
		s_t1 = random.choice(choices)
	#print s_t1
	return s_t1

def show_V( V ):
	#function to display V as a heat map
	#for row in V:
	#	print (row)
	plt.imshow(V, cmap='hot', interpolation='nearest')
	plt.show()
	return

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

if __name__ == '__main__':
	#[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
	discount = [0.35] #valid range: 0 <= discount < 1
	epsilion = [0.35] #valid range: 0 <= discount <= 1
	update_factor = [0.25] #valid range: 0 <= discount <= 1
	trace_decay = [0.7] #valid range: 0 <= discount <= 1
	iterations = []
	best = (None, None, None, None, 0, 0)
	grid = create_world(filename='grid1')
	#grid = create_world(density=0.1, do_print=False)
	#write_csv('grid4', grid)
	print ('discount, epsilion, update_factor, trace_decay, average_reward, average_steps')
	for i in range(10):
		for d in discount:
			for e in epsilion:
				for u in update_factor:
					for t in trace_decay:
						V, average_reward, average_steps = main(grid, d, e, u, t)
						run = [d, e, u, t, average_reward, average_steps]
						iterations.append(run[4:])
						if best[4] < run[4]:
							best = run
						print (run)
	print ('')
	print (best)
	rew = 0
	ste = 0
	for i in iterations:
		rew += i[0]
		ste += i[1]
	print rew/10.0, ste/10.0
	#write_csv('grid1_course', iterations)
	#show_V(V)



