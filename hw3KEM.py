import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib 

def get_inputs(filename):
    arrays = np.load(filename)
    return arrays['x1'], arrays['x2'], arrays['t1'], arrays['t2']

'''
x is the vector of inputs
D is the dimension of the vectors
K is the number of clusters
'''
def Kmeans(x, D, K, max_iter):
	#initialize the means to random values
	clusters = np.random.randn(K, D)
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

'''
x is the vector of inputs (N x D)
D is the dimension of the vectors
K is the number of clusters
sc is the stopping criterion
kmeans is the starting point for the means
'''
def EM(x, D, K, max_iter, sc, kmeans):
	N = len(x) #N is the number of training examples
	means = kmeans
	weights = np.random.randn(N, K)
	covs = np.zeros((K, D, D))
	for ki in range(K):
		covs[ki] = np.identity(D)
	mixings = np.full(K, 1.0/K)
	iter = 0
	logl = 0
	while(True):
		iter += 1
		
		#E step
		for ni in range(N):
			for ki in range(K):
				weights[ni][ki] = mixings[ki] * multivariate_normal.pdf(x[ni], mean= means[ki], cov = covs[ki])
			weights[ni] /= np.sum(weights[ni])

		Nk = np.zeros(K)
		for ki in range(K):
			Nk[ki] = np.sum(weights.T[ki])

		#M step
		mixings = np.divide(Nk,N)
		means = np.dot(weights.T, x)
		for ki in range(K):
			means[ki] = np.divide(means[ki], Nk[ki])

		for ki in range(K):
			covs[ki] = np.zeros((D, D))
			for ni in range(N):
				arr = np.array(x[ni] - means[ki]).reshape(D, 1)
				covs[ki] += np.dot(weights[ni][ki], np.dot(arr, arr.T))
			covs[ki] = np.divide(covs[ki], Nk[ki])

		#log-likelihood
		previous_logl = logl
		logl = 0
		for ni in range(N):
			loglk = 0
			for ki in range(K):
				loglk += mixings[ki] * multivariate_normal.pdf(x[ni], mean= means[ki], cov = covs[ki])
			logl += np.log(loglk)
		print "Log-Likelihood is " + str(logl)		
		print "Diff from previous LogL " + str(previous_logl - logl)
		if (np.absolute(previous_logl - logl) < sc):
			print "stopping criterion met after " + str(iter) + " iterations"
			break
		if (iter > max_iter):
			print "max iterations reached"
			break

	return means, covs



if __name__ == '__main__':
	x1, x2, t1, t2 = get_inputs("hw3_train.npz")
	x = np.concatenate((x1, x2), axis = 0)
	x = np.reshape(x,(np.size(x,0), (np.size(x,1)*np.size(x,2))))
	clusters = Kmeans(x, 36, 2 ,200)
	print "K-Means clusters: " + str(clusters)
	means, covs = EM(x, 36, 2, 200, 0.01, clusters)
	print "EM means: " + str(means)
	print "EM covariance matricies" + str(covs)
