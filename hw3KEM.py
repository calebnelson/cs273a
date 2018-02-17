import matplotlib.pyplot as plt
import numpy as np
import matplotlib 

def get_inputs(filename):
    arrays = np.load(filename)
    return arrays['x'], arrays['t'], arrays['classes']

'''
x is the vector of inputs
D is the dimension of the vectors
K is the number of clusters
'''
def Kmeans(x, D, K, max_iter):
	#initialize the means to random values
	clusters = np.random.randn(K, D)
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

'''
x is the vector of inputs
D is the dimension of the vectors
K is the number of clusters
sc is the stopping criterion
'''
def EM(x, D, K, max_iter, sc):
	N = len(x) #N is the number of training examples
	means = np.random.randn(K, D)
	covs = np.random.randn(K, D, D)
	mixings = np.full(K, 1.0/K)
	weights = np.random.randn(K, N)
	iter = 0
	while(True):
		iter += 1
		
		#E step
		for ni in range(N):
			for ki in range(K):
				weights[ki][ni] = mixings[ki] * multivariate_normal.pdf(x[ni], mean= means[ki], cov = covs[ki])
			weights[ki] /= np.sum(weights[ki])

		Nk = np.zeros(K)
		for ki in range(K):
			Nk[ki] = np.sum(weights[ki])

		#M step
		mixings = np.divide(Nk,N)
		means = np.dot(weights, x)
		for ki in range(K):
			means[ki] = np.divide(means[ki], Nk[ki])

		for ki in range(K):
			covs[ki] = np.zeros((D, D))
			for ni in range(N):
				arr = np.array(x[ni] - means[ki]).reshape(D, 1)
				covs[ki] += np.dot(weights[ki][ni], np.dot(arr, arr.T))
			covs[ki] = np.divide(covs[ki], Nk[ki])

		#log-likelihood
		previous_logl = logl
		logl = 0
		for ni in range(N):
			loglk = 0
			for ki in range(K):
				loglk += mixings[ki] * multivariate_normal.pdf(x[ni], mean= means[ki], cov = covs[ki])
			logl += np.log(loglk)
		if (np.absolute(previous_logl, logl) < sc):
			print "stopping criterion met after " + str(iter) + "iterations"
			break
		if (iter > max_iter):
			print "max iterations reached"
			break

	return means, covs



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
	
