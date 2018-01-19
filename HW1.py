import numpy as np
import matplotlib


def find_sw(seti, mi):
	swk = np.zeros((9,9))
	for row in seti:
		mean_distance = np.asmatrix(row - mi)
		swi = np.multiply(mean_distance, mean_distance.T)
		swk += swi
	return swk

def get_mean_of_sets(set1, set2):
	set_all = np.concatenate((set1, set2))
	m = np.zeros(9)
	for i in range(0, 9):
		m[i] = np.mean(set_all[:, i])
	return m

def find_bias(set1, set2, weights, m = np.zeros(9)):
	if (not m.any()):
		m = get_mean_of_sets(set1, set2)
	return np.dot(m, weights)

def fisher(set1, set2):
	m1 = np.empty(9)
	m2 = np.empty(9)
	for i in range(0, 9):
		m1[i] = np.mean(set1[:, i])
		m2[i] = np.mean(set2[:, i])

	sw1 = find_sw(set1, m1)
	sw2 = find_sw(set2, m2)
	sw = sw1 + sw2
	weights = np.dot(np.linalg.inv(sw), (m2-m1))
	return weights

def perceptron(set1, set2, learning_rate, p = True):
	m = get_mean_of_sets(set1, set2)
	N = len(set1) + len(set2)
	weights = np.zeros(9)
	bias = find_bias(set1, set2, weights, m)
	for epoch in range(200):
		tep = 0
		for i in range(N):
			if (i % 2 == 0):
				row = set1[i/2]
				x = (np.dot(row, weights))
				if (x - bias >= 0):
					for i in range(9):
						weights[i] -= row[i] * learning_rate
					bias = find_bias(set1, set2, weights, m)
					tep += x

			else:
				row = set2[i/2]
				x = (np.dot(row, weights))
				if (x - bias < 0):
					for i in range(9):
						weights[i] += row[i] * learning_rate
					bias = find_bias(set1, set2, weights, m)
					tep += x
		if(p):
			print "Total Energy Potential at Epoch " + str(epoch) + ": " + str(tep)
		if (tep == 0):
			break
	return weights

def test_weights(set1, set2, weights, p = True):
	N = len(set1) + len(set2)
	bias = find_bias(set1, set2, weights)
	ne1 = 0
	ne2 = 0
	for i in range(N):
		if (i % 2 == 0):
			row = set1[i/2]
			x = (np.dot(row, weights))
			if (x - bias >= 0):
				ne1 += 1

		else:
			row = set2[i/2]
			x = (np.dot(row, weights))
			if (x - bias < 0):
				ne2 += 1
	if(p):
		print "num errors after test1"
		print ne1
		print "num errors after test2"
		print ne2
		print "error rate:" + str((ne1+ne2)/N) + "\n"
	return (ne1+ne2)/N

def test_centers(ccenter1, ccenter2, alpha = 0.1, test_ratio = 0.1, learning_rate = 0.1, N = 10000):
	cov = np.identity(9) * alpha
	set1 = np.random.multivariate_normal(ccenter1, cov, N)
	set2 = np.random.multivariate_normal(ccenter2, cov, N)

	#Find the weights using just the fisher discriminant and
	fisher_weights = fisher(set1, set2) #1b
	#The weights using the perceptron learning algorithm
	perceptron_weights = perceptron(set1, set2, learning_rate) #1c

	#Intialize our test data
	set_test1 = np.random.multivariate_normal(ccenter1, cov, (int)(N*(test_ratio/2)))
	set_test2 = np.random.multivariate_normal(ccenter2, cov, (int)(N*(test_ratio/2)))
	
	#Test our weights
	print "Testing fisher weights"
	fisher_err = test_weights(set1, set2, fisher_weights)
	print "Testing perceptron weights"
	perceptron_err = test_weights(set1, set2, perceptron_weights)
	return (fisher_err, perceptron_err)

if __name__ == "__main__":
	alpha = 0.1
	test_ratio = .1
	learning_rate = 0.1
	N = 10000

	#Problem 1 using Gaussian distributions
	print "Testing Gaussian"
	ccenter1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
	ccenter2 = [3, 2.5, 1, 6, 4.5, 4, 9, 8.5, 7]
	test_centers(ccenter1, ccenter2)

	#Problem 2 using 3x3 binary images
	print "Testing Images"
	ccenter3 = [1, 0, 0, 0, 0, 0, 0, 1, 0]
	ccenter4 = [0, 0, 1, 0, 0, 1, 0, 0, 0]
	test_centers(ccenter3, ccenter4)

