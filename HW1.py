import numpy as np
import matplotlib

#Problem 1a
alpha = 1
ccenter1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
#ccenter2 = [5, 9, 2, 8, 3, 7, 4, 6, 1]
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

print "weights"
print (weights)

set_all = np.concatenate((set1, set2))
m = np.zeros(9)
for i in range(0, 9):
	m[i] = np.mean(set_all[:, i])
bias = np.dot(m, weights)
print "bias"
print bias

set_test1 = np.random.multivariate_normal(ccenter1, cov, N/4)
set_test2 = np.random.multivariate_normal(ccenter2, cov, N/4)
num_errors = 0.0
print "results for test set"
for row in set_test1:
	x = (np.dot(row, weights))
	if (x - bias > 0):
		num_errors +=1
for row in set_test2:
	x = (np.dot(row, weights))
	if (x - bias < 0):
		num_errors += 1
print "error rate:" + str(num_errors/(N/2))
