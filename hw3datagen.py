import matplotlib.pyplot as plt
import numpy as np

def generate_inputs(mean1, mean2, alpha, N):
    dim = np.size(mean1, axis=0)
    cov_mat = np.identity(dim) * alpha

    x1 = np.random.multivariate_normal(mean1, cov_mat, N)
    t1 = np.full((N, 2), [1, 0])
    x2 = np.random.multivariate_normal(mean2, cov_mat, N)
    t2 = np.full((N, 2), [0, 1])
    
    result_x1 = np.zeros((N, 6, 6))
    result_x2 = np.zeros((N, 6, 6))
    for i in range(N):
        result_x1[i] = np.copy(np.reshape(x1[i], (6, 6)))
        result_x2[i] = np.copy(np.reshape(x2[i], (6, 6)))
    
    return result_x1, result_x2, t1, t2

if __name__ == '__main__':
    mean1 = np.array(
    [0.5, 1, 1, 1, 1, .5, 
     0.5, 0.5, .5, .5, 1, .5,
     0.5, 1, 1, 1, 1, .5,
     0.5, 1, .5, .5, .5, .5,
     0.5, 1, 1, 1, 1, .5,
     0.5, 0.5, .5, .5, .5, .5
    ])
    mean2 = np.array(
    [0.5, 1, 1, 1, 1, .5, 
     0.5, 1, .5, .5, 0.5, .5,
     0.5, 1, 1, 1, 1, .5,
     0.5, 0.5, .5, .5, 1, .5,
     0.5, 1, 1, 1, 1, 0.5,
     0.5, 0.5, .5, .5, 0.5, .5
    ])

    classes = [0, 1]
    alpha = 0.01
    x1, x2, t1, t2 = generate_inputs(mean1, mean2, alpha, 2000)
    np.savez("hw3_train", x1=x1, x2=x2, t1=t1, t2=t2)
    x1, x2, t1, t2 = generate_inputs(mean1, mean2, alpha, 200)
    np.savez("hw3_test", x1=x1, x2=x2, t1=t1, t2=t2)
