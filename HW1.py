import numpy as np
import matplotlib

#Problem 1a
alpha = 1
ccenter1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
ccenter2 = [3, 2.5, 1, 6, 4.5, 4, 9, 8.5, 7]

N = 10000
cov = np.identity(9) * alpha
set1 = np.random.multivariate_normal(ccenter1, cov, N)
set2 = np.random.multivariate_normal(ccenter2, cov, N)

#Problem 1b
m1 = np.empty(9)
m2 = np.empty(9)
for i in range(0, 9):
	m1[i] = np.mean(set1[:, i])
	m2[i] = np.mean(set2[:, i])

def find_sw(seti, mi):
	swk = np.zeros((9,9))
	for row in seti:
		mean_distance = np.asmatrix(row - mi)
		swi = np.multiply(mean_distance, mean_distance.T)
		swk += swi
	return swk

sw1 = find_sw(set1, m1)
sw2 = find_sw(set2, m2)
sw = sw1 + sw2
weights = np.dot(np.linalg.inv(sw), (m2-m1))

set_all = np.concatenate((set1, set2))
m = np.zeros(9)
for i in range(0, 9):
	m[i] = np.mean(set_all[:, i])
bias = np.dot(m, weights)
print "bias"
print bias

print "weights before training"
print weights

#Problem 1c
test_ratio = .5
learning_rate = 0.01
set_test1 = np.random.multivariate_normal(ccenter1, cov, (int)(N*(test_ratio/2)))
set_test2 = np.random.multivariate_normal(ccenter2, cov, (int)(N*(test_ratio/2)))

ne1 = 0
ne2 = 0
print "results for test set"
for i in range((int)(N*test_ratio)):
	if (i % 2 == 0):
		row = set_test1[i/2]
		x = (np.dot(row, weights))
		if (x - bias >= 0):
			ne1 +=1
			for i in range(9):
				weights[i] -= row[i] * weights[i] * learning_rate

	else:
		row = set_test2[i/2]
		x = (np.dot(row, weights))
		if (x - bias < 0):
			ne2 += 1
			for i in range(9):
				weights[i] += row[i] * weights[i] * learning_rate
print "num errors after test1"
print ne1
print "num errors after test2"
print ne2
print "error rate:" + str((ne1+ne2)/(N*test_ratio))

bias = np.dot(m, weights)
print "\nnew bias"
print bias

print "new weights"
print weights

ne1 = 0
ne2 = 0
print "results for test set"
for i in range((int)(N*test_ratio)):
	if (i % 2 == 0):
		row = set_test1[i/2]
		x = (np.dot(row, weights))
		if (x - bias >= 0):
			ne1 +=1

	else:
		row = set_test2[i/2]
		x = (np.dot(row, weights))
		if (x - bias < 0):
			ne2 += 1
print "num errors after test1"
print ne1
print "num errors after test2"
print ne2
print "error rate:" + str((ne1+ne2)/(N*test_ratio))
