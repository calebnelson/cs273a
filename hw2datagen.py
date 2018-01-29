import numpy as np

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

if __name__ == "__main__":
    N = 10000
    alpha = 1.0
    cluster_center1 = [1, 2]
    class1_radius = 0.5
    noise = 0.1

    x_1a, t_1a, plot_classes_1a = create_inputs_1a(cluster_center1, alpha, N, class1_radius, noise)
    v_x_1a, v_t_1a, v_plot_classes_1a = create_inputs_1a(cluster_center1, alpha, N/5, class1_radius, noise)
    t_x_1a, t_t_1a, t_plot_classes_1a = create_inputs_1a(cluster_center1, alpha, N/3, class1_radius, noise)
    
    np.savez("hw2_train_1a", x=x_1a, t=t_1a, classes=plot_classes_1a)
    np.savez("hw2_validation_1a", x=v_x_1a, t=v_t_1a, classes=v_plot_classes_1a)
    np.savez("hw2_test_1a", x=t_x_1a, t=t_t_1a, classes=t_plot_classes_1a)
