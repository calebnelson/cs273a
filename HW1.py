import numpy as np
import matplotlib

alpha = 1
ccenter1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
ccenter2 = [5, 9, 2, 8, 3, 7, 4, 6, 1]

cov = np.identity(9) * alpha
set1 = np.random.multivariate_normal(ccenter1, cov, 9).T
print(len(set1))
print(set1)