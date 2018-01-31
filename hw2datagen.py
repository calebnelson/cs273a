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
        [[float]] means: an array of means of the gaussian distributions
        [[float]] classes : an array of classes of 0 or 1 that labels each cluster
        [[float]] alphas: an array the alphas of the gaussian distributions to generate for each cluster
        [[int]] Ns: an array the number of inputs vectors to generate for each cluster
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

def create_inputs_2(mean_1, mean_2, alpha, N):
    """create a number of input vectors from a gaussian distribution across a set amount of dimensions
        float mean: the mean of the gaussian distribution
        float alpha: the standard deviation of the gaussian distribution
        int N: number of input vectors to generate
        returns 
    """
    #generate input matrix by selecting random points from gaussian distribution
    dimensions = np.size(mean_1, axis=0)
    covariance_mat = np.identity(dimensions) * alpha

    input_mat_1 = np.random.multivariate_normal(mean_1, covariance_mat, N)
    training_classes_mat_1 = np.full((N, 2), [1, 0])
    matplot_classes_1 = np.zeros(N);

    input_mat_2 = np.random.multivariate_normal(mean_2, covariance_mat, N)
    training_classes_mat_2 = np.full((N, 2), [0, 1])
    matplot_classes_2 = np.ones(N);

    input_mat = np.concatenate((input_mat_1, input_mat_2), axis=0)
    training_classes_mat = np.concatenate((training_classes_mat_1, training_classes_mat_2), axis=0)
    matplot_classes = np.concatenate((matplot_classes_1, matplot_classes_2), axis=0)

    return input_mat, training_classes_mat, matplot_classes

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
    
    means = [[0, 0], [4, 4], [0, 4], [4, 0]]
    classes = [0, 0, 1, 1]
    alphas = np.full(4, 1.0)
    Ns1000 = np.full(4, 1000)
    Ns333 = np.full(4, 333)
    Ns200 = np.full(4, 200)
    x_1b, t_1b, plot_classes_1b = create_inputs_1b(means, classes, alphas, Ns1000)
    v_x_1b, v_t_1b, v_plot_classes_1b = create_inputs_1b(means, classes, alphas, Ns333)
    t_x_1b, t_t_1b, t_plot_classes_1b = create_inputs_1b(means, classes, alphas, Ns200)    
    np.savez("hw2_train_1b", x=x_1b, t=t_1b, classes=plot_classes_1b)
    np.savez("hw2_validation_1b", x=v_x_1b, t=v_t_1b, classes=v_plot_classes_1b)
    np.savez("hw2_test_1b", x=t_x_1b, t=t_t_1b, classes=t_plot_classes_1b)

    cluster_2 = [.1,.5,.5,.1,.1, .6,.1,.5,.5,.0, .1,.8,.9,.1,.0, .1,.0,.1,.1,.0, .2,.3,.4,.5,.6, .1,.5,.5,.1,.1, .6,.1,.5,.5,.0]
    cluster_1 = [.1,.4,.2,.1,.0, .6,.1,.3,.5,.0, .1,.8,.9,.0,.0, .1,.0,.1,.1,.0, .2,.3,.6,.5,.0, .1,.5,.4,.1,.8, .6,.1,.3,.5,.0]
    classes = [0, 1]
    alpha = 0.1
    N = 1000
    x_2, t_2, plot_classes_2 = create_inputs_2(cluster_1, cluster_2, alpha, N)
    v_x_2, v_t_2, v_plot_classes_2 = create_inputs_2(cluster_1, cluster_2, alpha, N/5)
    t_x_2, t_t_2, t_plot_classes_2 = create_inputs_2(cluster_1, cluster_2, alpha, N/3)
    np.savez("hw2_train_2", x=x_2, t=t_2, classes=plot_classes_2)
    np.savez("hw2_validation_2", x=v_x_2, t=v_t_2, classes=v_plot_classes_2)
    np.savez("hw2_test_2", x=t_x_2, t=t_t_2, classes=t_plot_classes_2)
