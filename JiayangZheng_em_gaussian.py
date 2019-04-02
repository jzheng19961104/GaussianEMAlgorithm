import numpy as np

DATA_PATH = ""

def parse_data(args):
    num = float
    dtype = np.float32
    data = []
    with open(args.data_file, 'r') as f:
        for line in f:
            data.append([num(t) for t in line.split()])
    dev_cutoff = int(.9*len(data))
    train_xs = np.asarray(data[:dev_cutoff],dtype=dtype)
    dev_xs = np.asarray(data[dev_cutoff:],dtype=dtype) if not args.nodev else None
    return train_xs, dev_xs

def init_model(args):
    clusters = []
    if args.cluster_num:
        # randomly initialize clusters (lambdas, mus, and sigmas)
        lambdas = np.zeros(args.cluster_num)
        lambdas.fill(1/args.cluster_num)
        mus = np.random.random((args.cluster_num,2))
        if not args.tied:
            sigmas = np.array([np.eye(2)]*args.cluster_num) #create an array of identity matrix
        else:
            sigmas = np.eye(2) #identity matrix

    else:
        lambdas = []
        mus = []
        sigmas = []
        with open(args.clusters_file,'r') as f:
            for line in f:
                #each line is a cluster, and looks like this:
                #lambda mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1
                lambda_k, mu_k_1, mu_k_2, sigma_k_0_0, sigma_k_0_1, sigma_k_1_0, sigma_k_1_1 = map(float,line.split())
                lambdas.append(lambda_k)
                mus.append([mu_k_1, mu_k_2])
                sigmas.append([[sigma_k_0_0, sigma_k_0_1], [sigma_k_1_0, sigma_k_1_1]])
        lambdas = np.asarray(lambdas)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(lambdas)

    #if args.tied was provided, sigmas will have a different shape
    #pack lambdas, mus, and sigmas in one array
    model = (lambdas,mus,sigmas)
    return model

def train_model(model, train_xs, dev_xs, tlist, dlist, args):
    from scipy.stats import multivariate_normal
    lambdas,mus,sigmas = extract_parameters(model)
    max_iter = args.iterations
    N = len(train_xs)
    K = args.cluster_num
    si = np.zeros((N,K))
    ll_old = 0

    for i in range(max_iter):

        #E-step
        for n in range(N):
            sum = 0.0
            for k in range(K):
                if not args.tied:
                    si[n,k] = lambdas[k] * multivariate_normal(mus[k], sigmas[k]).pdf(train_xs[n])
                else:
                    si[n,k] = lambdas[k] * multivariate_normal(mus[k], sigmas).pdf(train_xs[n])
                sum += si[n,k]
            for k in range(K):
                si[n,k] /= sum

        #M-step
        lambdas.fill(0)
        mus.fill(0)
        sigmas.fill(0)

        for k in range(K):

            for n in range(N):
                lambdas[k] += si[n,k]
            lambdas[k] /= N

            sum = 0.0
            for n in range(N):
                mus[k] += si[n,k]*train_xs[n]
                sum += si[n,k]
            mus[k] /= sum

            sum = 0.0
            if not args.tied:
                for n in range(N):
                    xn = np.reshape(train_xs[n]-mus[k],(2,1))
                    sigmas[k] += si[n,k]*np.dot(xn,xn.T)
                    sum += si[n,k]
                sigmas[k] /= sum

        #if covariance is tied, then sigma has a different shape than not tied
        if args.tied:
            for n in range(N):
                for k in range(K):
                    xn = np.reshape(train_xs[n]-mus[k],(2,1))
                    sigmas += si[n,k]*np.dot(xn,xn.T)
            sigmas /= N

        ll_train = average_log_likelihood(model,train_xs,args)
        tlist[i] = ll_train

        #convergence check
        if not args.nodev:
            ll_dev = average_log_likelihood(model, dev_xs, args)
            dlist[i] = ll_dev
            ll_new = ll_dev
            if ll_new-ll_old < 0.01:
                break
            ll_old = ll_new

    return model

def average_log_likelihood(model, data, args):
    from math import log
    from scipy.stats import multivariate_normal
    ll = 0.0
    lambdas,mus,sigmas = extract_parameters(model)
    clustern = len(mus)
    datasize = len(data)
    #sum over log likelihood of all data points
    for i in range(datasize):
        sum = 0.0
        #sum over all probability of clusters for a data point
        for j in range(clustern):
            if not args.tied:
                sum += lambdas[j] * multivariate_normal(mus[j], sigmas[j]).pdf(data[i])
            else:
                sum += lambdas[j] * multivariate_normal(mus[j], sigmas).pdf(data[i])
        ll += log(sum)

    ll /= datasize

    return ll

def extract_parameters(model):
    lambdas = model[0]
    mus = model[1]
    sigmas = model[2]
    return lambdas, mus, sigmas

def main():
    import argparse
    import os
    print('Gaussian') #Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points.')
    init_group = parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument('--cluster_num', type=int, help='Randomly initialize this many clusters.')
    init_group.add_argument('--clusters_file', type=str, help='Initialize clusters from this file.')
    parser.add_argument('--nodev', action='store_true', help='If provided, no dev data will be used.')
    parser.add_argument('--data_file', type=str, default=os.path.join(DATA_PATH, 'points.dat'), help='Data file.')
    parser.add_argument('--print_params', action='store_true', help='If provided, learned parameters will also be printed.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of EM iterations to perform')
    parser.add_argument('--tied',action='store_true',help='If provided, use a single covariance matrix for all clusters.')
    args = parser.parse_args()
    if args.tied and args.clusters_file:
        print('You don\'t have to (and should not) implement tied covariances when initializing from a file. Don\'t provide --tied and --clusters_file together.')
        exit(1)

    tlist = [0.0] * (args.iterations)
    dlist = [0.0] * (args.iterations)

    train_xs, dev_xs = parse_data(args)
    model = init_model(args)
    model = train_model(model, train_xs, dev_xs, tlist, dlist, args)
    ll_train = average_log_likelihood(model, train_xs, args)
    print('Train LL: {}'.format(ll_train))
    if not args.nodev:
        ll_dev = average_log_likelihood(model, dev_xs, args)
        print('Dev LL: {}'.format(ll_dev))
    lambdas, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str,a))
        print('Lambdas: {}'.format(intersperse(' | ')(np.nditer(lambdas))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '),mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '),map(lambda s: np.nditer(s),sigmas)))))

    #plot average log likelihood vs. num of iterations
    import matplotlib.pyplot as plt
    import matplotlib.patches as patch

    plt.plot(tlist, color='blue')
    train_patch = patch.Patch(color='blue', label='train data')

    if not args.nodev:
        plt.plot(dlist, color='green')
        development_patch = patch.Patch(color='green', label='development data')

    plt.title('Expectation Maximization with Cluster Number %s'%args.cluster_num)
    plt.ylabel('Average Log Likelihood')
    plt.xlabel('Num of iteration')
    plt.xlim(left=1)
    plt.xlim(right=(args.iterations + 1))
    plt.legend(handles=[train_patch, development_patch])
    plt.show()

if __name__ == '__main__':
    main()