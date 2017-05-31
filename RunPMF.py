import sys
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy import sparse
import scipy
#import seaborn as sns
from PMF import *
import math
#sns.set()
#np.random.seed(0)

def DCG(truth):
    n = len(truth)
    res = 0
    for i in range(n):
        res += (2**truth[i] - 1.) / (math.log(i+2., 2.))
    return float(res) / n

def NDCG(truth):
    try:
        dcg = DCG(truth)
        perfect_DCG = DCG(sorted(truth, key=lambda x: -x))
        return dcg / perfect_DCG
    except:
        return  0.


def save_sparse_matrix(filename, x):
    x_coo = x.tocoo()
    row = x_coo.row
    col = x_coo.col
    data = x_coo.data
    shape = x_coo.shape
    np.savez(filename, row=row, col=col, data=data, shape=shape)
    f = open(filename, 'w')
    for i in range(row.shape[0]):
        f.write(str(row[i]) + "," + str(col[i]) + "," + str(data[i]) + "\n")

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
        if len(ratings[user,:].nonzero()[0]) < 15:
            continue
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
    names = ['userId', 'movieId', 'rating', 'timestamp']
    df = pd.read_csv(directory + "/" + 'ratings3.csv', names=names)
    #df = pd.read_csv(directory + "/" + 'ratings_negative1.csv', names=names)
    n_users = max(df.userId.unique() + 1)
    n_items = max(df.movieId.unique() + 1)
    #print(n_users, n_items)
    #ratings = np.zeros((n_users, n_items))
    ratings = lil_matrix((n_users, n_items))
    print(ratings.shape)
    for row in df.itertuples():
        #print(row[1], row[2], row[3])
        ratings[row[1], row[2]] = (int(row[3] > 3.5) - 0.5) * 2
        #ratings[row[1], row[2]] = (int(row[3] > 0.5) - 0.5) * 2



    print str(n_users) + ' users'
    print str(n_items) + ' items'
    sparsity = float(len(ratings.nonzero()[0]))
    print(sparsity, n_items, n_users)
    sparsity /= (ratings.shape[0] * ratings.shape[1])
    sparsity *= 100
    print 'Sparsity: {:4.2f}%'.format(sparsity)
    return ratings

def GetTestUsers(ratings, n_users_in_test, dir):
    
    #print(ratings.shape)
    (x,y,z) = scipy.sparse.find(ratings)
    #countings=np.bincount(x)
    #users = np.where(countings>10)[0]
    #ratings = ratings[users]
   
    (x,y,z) = scipy.sparse.find(ratings)
    countings=np.bincount(x)
    #users_bias = np.mean(ratings, axis=1)
    #users = np.where(abs(users_bias) < 0.33)[0]
    
    #ratings = ratings[users]
    print(ratings.shape)
    

    array = np.arange(ratings.shape[0])
    np.random.shuffle(array)
    test_users = array[:n_users_in_test]

    # filter items
    ratings1 = ratings[array[n_users_in_test:]].T
     
    (x, y, z) = sparse.find(ratings1)
    countings = np.bincount(x)
    sums = np.bincount(x, weights=z)
    averages = sums / (countings +1e-10)
    print(averages)

    items = []
    for i in xrange(ratings1.shape[0]):
        # print(ratings1[i].nonzero(), len(ratings1[i].nonzero()[0]))
        if len(ratings1[i].nonzero()[1]) > 10:
            items.append(i)
   
    
    (x, y, z) = sparse.find(ratings1)
    countings = np.bincount(x)
    print(len(items))
    ratings = ratings.T[items]
    ratings = ratings.T
    np.savetxt(dir  + "test_items.txt", items)
    return ratings[array[n_users_in_test:]], ratings[test_users], array

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
    ndcg = 0.
    n_positive_ex = 0
    for user in xrange(ratings.shape[0]):
        user_result = []
        truth = []
        test_ratings = ratings[user].nonzero()[0]
        for item in test_ratings:
            #print(user, item, ratings[user, item])
            #print(model.predict(user, item))
            if ratings[user,item] > 0:
                n_positive_ex += 1
            result.append([ratings[user, item], model.predict(user, item), user, item, model.item_bias[item], model.user_bias[user], ','.join(str(u) for u in model.user_vecs[user])])
            user_result.append([ratings[user, item], model.predict(user, item), user, item])
        user_result.sort(key = lambda x:(-x[-1],-x[1]))
        for r in user_result:
            truth.append(r[0])
        ndcg1 = NDCG(np.array(truth) + 1.)
        ndcg += ndcg1
        if (ndcg1 > 1.):
            print (truth, ndcg1)
        #print (ndcg/(user + 1.))


    result.sort(key = lambda x:-x[1])
    n_positive_local = 0
    error = 0.
    i = 0.
    with open('res.txt', 'w') as res:
        for r in result:
            i += 1
            mse += (r[1] - r[0]) ** 2
            if r[0] > 0:
                n_positive_local += 1
            local_error = (n_positive_local + len(result) - i - n_positive_ex + n_positive_local) / len(result)
            if (local_error > error):
                error = local_error
           # error += int((r[1] * r[0]) > 0)
            n_ex += 1
            res.write(str(r[0]) + "\t" + str(r[1]) + "\t" + str(r[2]) +"\t" + str(r[3]) + "\t" + str(r[4]) + "\t" + str(r[5]) + "\t" + str(r[6])+'\n')
    return error, mse / n_ex

def main(n_factors, i):
    print(n_factors)
    #dir_with_data = "../../movieLens/ml-100k"
    dir_with_data = "../DATA/ml-20m"
    dir = "../PWL/data" + str(i) + "/"
    #train, test1 = train_test_split(GetData1(dir_with_data))
    ratings = GetData1(dir_with_data)
    n_users_in_test = 1000#ratings.shape[0] / 3
    train, test, users_order = GetTestUsers(ratings, n_users_in_test, dir)
    best_sgd_model = ExplicitMF(train, n_factors=n_factors, learning='sgd', \
                            item_fact_reg=0.01, user_fact_reg=0.01, \
                            user_bias_reg=0.01, item_bias_reg=0.01)
    save_sparse_matrix(dir + "tes_ratings.txt", test)

    #np.savetxt(dir + "tes_ratings.txt", test.data)
    for i in range(2):
        best_sgd_model.train(100, learning_rate=0.01 / (i + 1), from_scratch=(i==0))
        #print(Print_result(best_sgd_model, test))

        np.savetxt(dir + "items.txt", best_sgd_model.item_vecs)
        print(best_sgd_model.item_vecs.shape)
        np.savetxt(dir + "items_bias.txt", best_sgd_model.item_bias)
        np.savetxt(dir + "users_train.txt", best_sgd_model.user_vecs)
        np.savetxt(dir + "user_bias_train.txt", best_sgd_model.user_bias)
        with open(dir + "global_bias.txt", 'w') as global_bias:
            global_bias.write(str(best_sgd_model.global_bias))

    best_sgd_model1 = ExplicitMF(test, n_factors=n_factors, learning='sgd', \
                                item_fact_reg=0.01, user_fact_reg=0.01, \
                                user_bias_reg=0.01, item_bias_reg=0.01)
    best_sgd_model1.item_vecs = best_sgd_model.item_vecs
    best_sgd_model1.item_bias = best_sgd_model.item_bias
    best_sgd_model1.global_bias = best_sgd_model.global_bias
    best_sgd_model1.train(200, learning_rate=0.01, from_scratch=False, user_step = True, item_step = False)
    print(Print_result(best_sgd_model1, test))
    np.savetxt(dir + "user_bias.txt", best_sgd_model1.user_bias)
    np.savetxt(dir + "users.txt", best_sgd_model1.user_vecs)
    np.savetxt(dir + 'test.txt', users_order)
#    with open(dir + "n_users_in_test.txt", 'w') as global_bias:
#        global_bias.write(str(n_users_in_test))
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

FLAGS = None
import argparse
if __name__ == '__main__':
    for i in range(1):
        parser = argparse.ArgumentParser()
        parser.add_argument('--f', type = str, default = 100)
        FLAGS, unparsed = parser.parse_known_args()
        main(int(FLAGS.f), i)
