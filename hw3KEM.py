import matplotlib.pyplot as plt
import numpy as np
import matplotlib 

def get_inputs(filename):
    arrays = np.load(filename)
    return arrays['x'], arrays['t'], arrays['classes']

'''
x is the vector of inputs
N is the dimension of the vectors
K is the number of clusters
'''
def Kmeans(x. N, K):
	#initialize the means to random values
	clusters = np.random.randn(N, K)/np.sqrt(N)
	cluster_assignments = [[] for i in range(K)]
	while(True):
		#assignment step
		iter += 1
		previous_ca = cluster_assignments
		cluster_assignments = [[] for i in range(K)]
		for point in x:
			distances = [numpy.linalg.norm(point - cluster) for cluster in clusters]
			cluster_assignments[np.argmin(distances)].append(point)
		#if cluster assignment is same as last time, we've converged
		if(np.array_equal(previous_ca, cluster_assignments)):
			print "Converged after " + str(iter) + " iterations"
			break

		#relocation step
		for i in range(K):
			a = np.array(cluster_assignments[i])
			clusters[i] = np.mean(a, axis=1)
	return clusters

if __name__ == '__main__':

	x_2, t_2, plot_classes_2 = get_inputs("hw2_train_2.npz")
	clusters = Kmeans(x_2, 35, 2)
	print clusters
	