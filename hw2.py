import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def create_inputs_1a(mean, alpha = 1, N = 100, class1_radius = 1, noise = 0.1):
    """create a number of input vectors from a gaussian distribution across a set amount of dimensions
        float mean: the mean of the gaussian distribution
        float alpha: the standard deviation of the gaussian distribution
        int N: number of input vectors to generate
    """
    #generate input matrix by selecting random points from gaussian distribution
    dimensions = np.size(mean, axis=0)
    covariance_mat = np.identity(dimensions) * alpha

    input_mat = np.random.multivariate_normal(mean, covariance_mat, N)
    training_classes_mat = np.full((N, 2), [1, 0])
    classes = np.zeros(N);

    for i in range(len(input_mat)):
        dist = np.linalg.norm(input_mat[i] - mean + noise * np.random.randn(dimensions))
        
        if dist > class1_radius:
            training_classes_mat[i] = [0, 1]
            classes[i] = 1
        else:
            training_classes_mat[i] = [1, 0]

    return input_mat, training_classes_mat, classes

def create_inputs_1b(means, classes, alphas, Ns):
    """create a number of input vectors from a gaussian distribution across a set amount of dimensions
        [[float]] mean: the mean of the gaussian distribution
        [[float]] alpha: the standard deviation of the gaussian distribution
        int N: number of input vectors to generate
    """
    #generate input matrix by selecting random points from gaussian distribution
    dimensions = np.size(means, axis=1)

    covariance_mat = np.identity(dimensions) * alpha
    input_mat = np.random.multivariate_normal(means[0], covariance_mat, Ns[0])
    classes_mat = np.full(Ns[0], classes[0])
    if (classes[0] == 0):
        training_classes_mat = np.full((Ns[0], 2), [1, 0])
    else:
        training_classes_mat = np.full((Ns[0], 2), [0, 1])

    for group in range(len(means) - 1):
        covariance_mat = np.identity(dimensions) * alpha
        new_input_mat = np.random.multivariate_normal(means[group + 1], covariance_mat, Ns[group + 1])
        if (classes[group + 1] == 0):
            new_training_classes_mat = np.full((Ns[0], 2), [1, 0])
        else:
            new_training_classes_mat = np.full((Ns[0], 2), [0, 1])

        input_mat = np.concatenate((input_mat, new_input_mat), axis=0)
        classes_mat = np.concatenate((classes_mat, np.full(Ns[group + 1], classes[group + 1])), axis=0)
        training_classes_mat = np.concatenate((training_classes_mat, new_training_classes_mat), axis=0)

    return input_mat, training_classes_mat, classes_mat
'''
x is the vector of inputs
t is the probabilities of the target vector
N is the dimension of the vectors
M is the dimension of the hidden layer
K is the number of classes
learning_rate is the step size of the gradient descend
eps is the stopping criteria
'''
def build_model(x, t, v_x, v_t, N, M, K, learning_rate, eps):
    #declare and initialize the 2 weight matrices and bias vectors
    np.random.seed(0)
    W1 = np.random.randn(N,M)/np.sqrt(N)
    b1 = np.zeros((1,M))
    W2 = np.random.randn(M,K)/np.sqrt(M)
    b2 = np.zeros((1,K))
    iter = 0
    prev_total_cross_entropy = 0
    total_cross_entropy = 0
    while (True):
        prev_total_cross_entropy = total_cross_entropy
        total_cross_entropy = 0
        iter += 1
        #start training, break if the target accuracy has be achieved
        for i in range(0, len(x)):
            #print('x', x)
            #print('w1', W1)
            #print('w2', W2)
            #print('t:', t)

            a1 = np.dot(x[i], W1)+b1
            #print('a1', a1)
            z1 = np.tanh(a1)
            #print('z', z1)
            a2 = np.dot(z1, W2)+b2
            #print('a2', a2)
            exp_scores = np.exp(a2)
            z2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            #print('z2', z2)
            delta_k = z2-t[i]
            #print('delta_k', delta_k)
            delta_j = (1 - np.power(z1, 2)) * (np.dot(delta_k, W2.T))
            #print('delta_j', delta_j)

            #calculate gradients
            X = np.asmatrix(x[i])
            dE1 = np.multiply(delta_j, X.T)
            db1 = np.sum(delta_j)
            Z = np.asmatrix(z1)
            dE2 = np.multiply(delta_k, Z.T)
            db2 = np.sum(delta_k)

            #update the weight vectors and biases
            W1 -= learning_rate*dE1
            W2 -= learning_rate*dE2
            b1 -= learning_rate*db1
            b2 -= learning_rate*db2

            #print('np.log(z2):', np.log(z2))
            #print('np.log(1-z2):', np.log(1-z2))
            #print('np.dot(t, np.log(z2)):', np.dot(t, np.log(z2[0])))
            #print('np.dot((1-t), np.log(1-z2)):', np.dot((1-t), np.log(1-z2[0])))
            cross_entropy = np.dot(t[i], np.log(z2[0])) + np.dot((1-t[i]), np.log(1-z2[0]))
            total_cross_entropy -= cross_entropy

        print('iteration: ', iter, 'cross_entropy: ', total_cross_entropy)
        if(abs(total_cross_entropy - prev_total_cross_entropy) < eps):
            break

    return (W1, W2, b1, b2)

def test_model(W1, W2, b1, b2, x, t):
    return 0


if __name__ == "__main__":
    
    N = 200
    alpha = 1.0
    learning_rate = 0.1

    cluster_center1 = [1, 2]
    class1_radius = 0.5
    noise = 0.1
    x_1a, t_1a, plot_classes_1a = create_inputs_1a(cluster_center1, alpha, N, class1_radius, noise)
    v_x_1a, v_t_1a, v_plot_classes_1a = create_inputs_1a(cluster_center1, alpha, N/5, class1_radius, noise)
    t_x_1a, t_t_1a, t_plot_classes_1a = create_inputs_1a(cluster_center1, alpha, N/3, class1_radius, noise)

    #print t_1a
    (W1, W2, b1, b2) = build_model(x_1a, t_1a, v_x_1a,  v_t_1a,  2,10,2, learning_rate, 0.01)

