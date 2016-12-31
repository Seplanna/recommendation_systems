import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from PMF import *
#sns.set()
np.random.seed(0)


def GetData(file):
    data = np.zeros(shape=(80000,3))
    line_index = 0
    with open(file, 'r') as f:
        for line in f:
            line = line.strip().split()
            data[line_index] = [int(line[0]), int(line[1]), int(int(line[2]) > 3)]
            line_index += 1
    dt = np.dtype('int,int,int')
    return data


def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in xrange(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                        size=10, 
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
        
    # Test and training are truly disjoint
    assert(np.all((train * test) == 0)) 
    return train, test

#def train_test_split_by_user(ratings, users):

def GetData1(directory):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(directory + "/" + 'u.data', sep='\t', names=names)
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    ratings = np.zeros((n_users, n_items))
    for row in df.itertuples():
        ratings[row[1]-1, row[2]-1] = (int(row[3] > 3.5) - 0.5) * 2
    print str(n_users) + ' users'
    print str(n_items) + ' items'
    sparsity = float(len(ratings.nonzero()[0]))
    sparsity /= (ratings.shape[0] * ratings.shape[1])
    sparsity *= 100
    print 'Sparsity: {:4.2f}%'.format(sparsity)
    return ratings

def plot_learning_curve(iter_array, model, save_fig):
    fig = plt.figure()
    plt.plot(iter_array, model.train_mse, \
             label='Training', linewidth=5)
    plt.plot(iter_array, model.test_mse, \
             label='Test', linewidth=5)
    plt.xticks(fontsize=16);
    plt.yticks(fontsize=16);
    plt.xlabel('iterations', fontsize=30);
    plt.ylabel('MSE', fontsize=30);
    plt.legend(loc='best', fontsize=20)
    fig.savefig(save_fig)

def OptimizeParametersSGD(train, test):
    iter_array = [1, 2, 5, 10, 25, 50, 100, 200]
    latent_factors = [5, 10, 20, 40, 80]
    regularizations = [0.001, 0.01, 0.1, 1.]
    regularizations.sort()
    best_params = {}
    best_params['n_factors'] = latent_factors[0]
    best_params['reg'] = regularizations[0]
    best_params['n_iter'] = 0
    best_params['train_mse'] = np.inf
    best_params['test_mse'] = np.inf
    best_params['model'] = None

    for fact in latent_factors:
        print 'Factors: {}'.format(fact)
        for reg in regularizations:
            print 'Regularization: {}'.format(reg)
            MF_SGD = ExplicitMF(train, n_factors=fact, learning='sgd',\
                            user_fact_reg=reg, item_fact_reg=reg, \
                            user_bias_reg=reg, item_bias_reg=reg)
            MF_SGD.calculate_learning_curve(iter_array, test, learning_rate=0.001)
            min_idx = np.argmin(MF_SGD.test_mse)
            if MF_SGD.test_mse[min_idx] < best_params['test_mse']:
                best_params['n_factors'] = fact
                best_params['reg'] = reg
                best_params['n_iter'] = iter_array[min_idx]
                best_params['train_mse'] = MF_SGD.train_mse[min_idx]
                best_params['test_mse'] = MF_SGD.test_mse[min_idx]
                best_params['model'] = MF_SGD
                print 'New optimal hyperparameters'
    print pd.Series(best_params)

def Print_result(model, ratings):
    error = 0.
    mse = 0
    n_ex = 0
    result = []
    for user in xrange(ratings.shape[0]):
        test_ratings = ratings[user, :].nonzero()[0]
        for item in test_ratings:
            result.append([ratings[user, item], model.predict(user, item)])
    result.sort(key = lambda x:x[1])
    with open('res.txt', 'w') as res:
        for r in result:
            mse += (r[1] - r[0]) ** 2
            error += int((r[1] * r[0]) > 0)
            n_ex += 1
            res.write(str(r[0]) + "\t" + str(r[1]) + '\n')
    return error / n_ex, mse / n_ex

def main():
    dir_with_data = "../../movieLens/ml-100k"
    train, test = train_test_split(GetData1(dir_with_data))
    ratings = GetData1(dir_with_data)
    best_sgd_model = ExplicitMF(ratings, n_factors=10, learning='sgd', \
                            item_fact_reg=0.01, user_fact_reg=0.01, \
                            user_bias_reg=0.01, item_bias_reg=0.01)
    for i in range(20):
        best_sgd_model.train(300, learning_rate=0.01 / (i + 1), from_scratch=(i==0))
        print(Print_result(best_sgd_model, test))
        dir = "../../RL/data/"
        np.savetxt(dir + "items.txt", best_sgd_model.item_vecs)
        np.savetxt(dir + "items_bias.txt", best_sgd_model.item_bias)
        np.savetxt(dir + "users.txt", best_sgd_model.user_vecs)
        np.savetxt(dir + "user_bias.txt", best_sgd_model.user_bias)
        with open(dir + "global_bias.txt", 'w') as global_bias:
            global_bias.write(str(best_sgd_model.global_bias))
#    best_sgd_model.item_vecs = np.genfromtxt(sys.argv[2])
#    best_sgd_model.item_bias = np.genfromtxt(sys.argv[3]) 
#    best_sgd_model.user_vecs = np.genfromtxt(sys.argv[4])
#    best_sgd_model.user_bias = np.genfromtxt(sys.argv[5])
#    with open(sys.argv[6], 'r') as global_bias:
#        for line in global_bias:
#            best_sgd_model.global_bias = float(line.strip())



def GetItemsNames(file):
    items_names = {}
    with open(file) as names:
        for line in names:
            line = line.strip().split['|']
            items_names[int(line[0])] = line[1]
    return items_names
    

main()
