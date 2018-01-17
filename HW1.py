import numpy as np
import matplotlib

#Problem 1a
alpha = 1
ccenter1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
ccenter2 = [5, 9, 2, 8, 3, 7, 4, 6, 1]

N = 100
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

for row in set1:
	print (np.dot(row, weights))
for row in set2:
	print (np.dot(row, weights))
