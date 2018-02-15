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
def Kmeans(x, N, K, max_iter):
	#initialize the means to random values
	clusters = np.random.randn(K, N)/np.sqrt(K)
	iter = 0
	while(True):
		cluster_assignments = [[] for i in range(K)]
		#assignment step
		iter += 1
		for point in x:
			distances = [np.linalg.norm(point - cluster) for cluster in clusters]
			cluster_assignments[np.argmin(distances)].append(point)

		#relocation step
		previous_clusters = np.copy(clusters)
		for i in range(K):
			#print len(cluster_assignments[i])
			a = np.array(cluster_assignments[i])
			clusters[i] = np.mean(a, axis=0)

		#if clusters are the same as last time, we've converged
		if(np.array_equal(previous_clusters, clusters)):
			print "Converged after " + str(iter) + " iterations"
			break
		if(iter >= max_iter):
			print "Hit max iterations"
			break	

		#print "After iter " + str(iter)
	return clusters

if __name__ == '__main__':

	x_1a, t_1a, plot_classes_1a = get_inputs("hw2_train_1a.npz")
	clusters = Kmeans(x_1a, 2, 2, 200)
	print clusters

	x_1b, t_1b, plot_classes_1b = get_inputs("hw2_train_1b.npz")
	clusters = Kmeans(x_1b, 2, 2, 200)
	print clusters

	x_2, t_2, plot_classes_2 = get_inputs("hw2_train_2.npz")
	clusters = Kmeans(x_2, 35, 2, 200)
	print clusters
	
