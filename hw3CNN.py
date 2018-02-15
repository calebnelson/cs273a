import matplotlib.pyplot as plt
import numpy as np
import matplotlib 

def get_inputs(filename):
    arrays = np.load(filename)
    return arrays['x'], arrays['t'], arrays['classes']

'''
x is the vector of inputs
t is the probabilities of the target vector
N is the dimension of the vectors
M is the dimension of the hidden layer
K is the number of classes
learning_rate is the step size of the gradient descend
eps is the stopping criteria
'''
def build_model(x, t, v_x, v_t, N, M, K, learning_rate, eps, target_accuracy, target_iter):
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
        for i in np.random.permutation(len(x)):
            a1 = np.dot(x[i], W1)+b1
            z1 = np.tanh(a1)
            a2 = np.dot(z1, W2)+b2
            exp_scores = np.exp(a2)
            z2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            delta_k = z2-t[i]
            delta_j = (1 - np.power(z1, 2)) * (np.dot(delta_k, W2.T))

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

        for i in range(0, len(v_x)):
            a1 = np.dot(v_x[i], W1)+b1
            z1 = np.tanh(a1)
            a2 = np.dot(z1, W2)+b2
            exp_scores = np.exp(a2)
            z2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            cross_entropy = np.dot(v_t[i], np.log(z2[0])) + np.dot((1-v_t[i]), np.log(1-z2[0]))
            total_cross_entropy -= cross_entropy

        error_rate = test_model(W1, W2, b1, b2, v_x, v_t)
        
        print('iteration: ', iter, 'cross_entropy: ', total_cross_entropy, ' error rate: ', error_rate)
        if(abs(total_cross_entropy - prev_total_cross_entropy) < eps):
            break
        elif error_rate < target_accuracy:
            print('early exit: target_accuracy met')
            break
        elif iter > target_iter:
            print('early exit: max_iter exceeded')
            break

    return (W1, W2, b1, b2)

def test_model(W1, W2, b1, b2, x, t):
    a1 = np.dot(x, W1)+b1
    z1 = np.tanh(a1)
    a2 = np.dot(z1, W2)+b2
    exp_scores = np.exp(a2)
    z2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    rounded_z2 = np.around(z2).astype(int) #round to [0, 1] or [1, 0]
    num_errors = 0.0
    for i in range(len(t)):
	if (not np.all(rounded_z2[i] == t[i])):
		num_errors += 1
    return (num_errors/len(t))

if __name__ == "__main__":
    
    learning_rate = 0.001
    epsilon = 0.01
    target_accuracy = 0.025
    target_iter = 100
    neurons = 10

    x_1a, t_1a, plot_classes_1a = get_inputs("hw2_train_1a.npz")
    v_x_1a, v_t_1a, v_plot_classes_1a = get_inputs("hw2_validation_1a.npz")
    t_x_1a, t_t_1a, t_plot_classes_1a = get_inputs("hw2_test_1a.npz")

    x_1b, t_1b, plot_classes_1b = get_inputs("hw2_train_1b.npz")
    v_x_1b, v_t_1b, v_plot_classes_1b = get_inputs("hw2_validation_1b.npz")
    t_x_1b, t_t_1b, t_plot_classes_1b = get_inputs("hw2_test_1b.npz")

    x_2, t_2, plot_classes_2 = get_inputs("hw2_train_2.npz")
    v_x_2, v_t_2, v_plot_classes_2 = get_inputs("hw2_validation_2.npz")
    t_x_2, t_t_2, t_plot_classes_2 = get_inputs("hw2_test_2.npz")

    (W1, W2, b1, b2) = build_model(x_1a, t_1a, v_x_1a,  v_t_1a,  2,10,2, learning_rate, epsilon, target_accuracy, target_iter)
    plt.figure()
    plt.imshow(W1, cmap='hot')
    plt.figure()
    plt.imshow(W2, cmap='hot')
    plt.figure()
    plt.scatter(x=x_1a[:,0], y=x_1a[:,1], s=20, c=plot_classes_1a)
    print("Error rate for 1a: " + str(test_model(W1, W2, b1, b2, t_x_1a, t_t_1a)*100) + "%")

    (W1, W2, b1, b2) = build_model(x_1b, t_1b, v_x_1b,  v_t_1b,  2,10,2, learning_rate, epsilon, target_accuracy, target_iter)
    plt.figure()
    plt.imshow(W1, cmap='hot')
    plt.figure()
    plt.imshow(W2, cmap='hot')
    plt.figure();
    plt.scatter(x=x_1b[:,0], y=x_1b[:,1], s=20, c=plot_classes_1b)
    print("Error rate for 1b: " + str(test_model(W1, W2, b1, b2, t_x_1b, t_t_1b)*100) + "%")


    cluster_2 = [.1,.5,.5,.1,.1, .6,.1,.5,.5,.0, .1,.8,.9,.1,.0, .1,.0,.1,.1,.0, .2,.3,.4,.5,.6, .1,.5,.5,.1,.1, .6,.1,.5,.5,.0]
    cluster_1 = [.1,.4,.2,.1,.0, .6,.1,.3,.5,.0, .1,.8,.9,.0,.0, .1,.0,.1,.1,.0, .2,.3,.6,.5,.0, .1,.5,.4,.1,.8, .6,.1,.3,.5,.0]
    (W1, W2, b1, b2) = build_model(x_2, t_2, v_x_2,  v_t_2, 35,neurons,2, learning_rate, epsilon, target_accuracy, target_iter)
    plt.figure()
    plt.imshow(W1, cmap='hot')
    plt.figure()
    plt.imshow(W2, cmap='hot')
    plt.figure();
    plt.imshow(np.reshape(cluster_1, (5,7)))
    plt.figure();
    plt.imshow(np.reshape(cluster_2, (5,7)))

    print("Error rate for 2: " + str(test_model(W1, W2, b1, b2, t_x_2, t_t_2)*100) + "%")
    plt.show()
