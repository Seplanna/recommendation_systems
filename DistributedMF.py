from RunPMF import GetTestUsers
from RunPMF import GetData1
from RunPMF import save_sparse_matrix
from RunPMF import train_test_split
from RunPMF import Print_result
from PMF import *
import pandas as pd
import argparse
from math import sqrt

parser = argparse.ArgumentParser()
parser.add_argument('--i', type = int)
parser.add_argument('--u', type = int)
parser.add_argument('--f', type = int)
FLAGS, unparsed = parser.parse_known_args()

data_dir = "../DATA/ml-20m"
dir = "../GeneratedData/data" + str(FLAGS.f) + "/"
item_dir = "../GeneratedData/data" + str(FLAGS.f) + "/ITEMS/"
user_dir = "../GeneratedData/data" + str(FLAGS.f) + "/USERS/"

n_test_users = 1000 
n_factors = 50
n_process = 30

def GetPartOFRating(users, items):
    names = ['userId', 'movieId', 'rating', 'timestamp']
    #df = pd.read_csv(data_dir + "/" + 'ratings.dat', sep='::', names=names)
    df = pd.read_csv(data_dir + "/" + 'ratings.csv')
    n_users = len(users.keys())
    n_items = len(items.keys())
    print(n_users, n_items)
    ratings = np.zeros((n_users, n_items))
    #ratings = lil_matrix((n_users, n_items))
    print(ratings.shape)
    for row in df.itertuples():
        if row[1] - 1 in users and row[2] - 1 in items:
            ratings[users[row[1] - 1], items[row[2] - 1]] = (int(row[3] > 3.5) - 0.5) * 2
    return ratings

def GetPartOfSparceRatings(users, items):
    test_ratings = np.load(dir + "train_ratings.txt.npz")
    data = test_ratings['data']
    col = test_ratings['col']
    rows = test_ratings['row']
    n_users = len(users.keys())
    n_items = len(items.keys())
    print(n_users, n_items)
    ratings = np.zeros((n_users, n_items))

    for i in range(len(rows)):
        if rows[i] in users and col[i] in items:
            ratings[users[rows[i]], items[col[i]]] = data[i]
    return ratings



def Get_items(items_array, n_process, process, del_ = 0):
    item_step = int(len(items_array) / n_process)
    items_ = items_array[item_step * process: item_step * (process+1)]
    if (del_ > 0):
        items_ = items_array[item_step * 0: item_step * (4)]

    res = {}
    for i in xrange(items_.shape[0]):
        res[items_[i]] = i
    return res

def GetDataDistr(item_set, user_set):
    users_array = np.genfromtxt("")
    items_array = np.genfromtxt("")
    item_vecs_file = item_dir + "vecs/" + str(item_set)
    item_bias_file = item_dir + "bias/" + str(item_set)
    user_vecs_file = user_dir + "vecs/" + str(user_set)
    user_bias_file = user_dir + "bias/" + str(user_set)
    item_vecs = np.genfromtxt(item_vecs_file)
    user_vecs = np.genfromtxt(user_vecs_file)
    item_bias = np.genfromtxt(item_bias_file)
    user_bias = np.genfromtxt(user_bias_file)
    return item_vecs, item_bias, user_vecs, user_bias

def RunOneProcess(user_set, item_set, n_process, n_factors = 100):
    #dir = ''
    #item_dir = ''
    #user_dir = ''

    users_array = np.genfromtxt(dir + "users_array.txt")
    items_array = np.genfromtxt(dir + "items_array.txt")
    item_vecs_file = item_dir + "vecs/" + str(item_set)
    item_bias_file = item_dir + "bias/" + str(item_set)
    user_vecs_file = user_dir + "vecs/" + str(user_set)
    user_bias_file = user_dir + "bias/" + str(user_set)
    item_vecs = np.genfromtxt(item_vecs_file)

    """for i in range(4):
        if (i == 0):
            item_vecs = np.genfromtxt(item_dir + "vecs/" + str(i))
            item_bias = np.genfromtxt(item_dir + "bias/" + str(i))
        else:
            iv = np.genfromtxt(item_dir + "vecs/" + str(i))
            ib = np.genfromtxt(item_dir + "bias/" + str(i))
            print(item_vecs.shape, item_bias.shape)
            item_vecs = np.concatenate((item_vecs, iv))
            item_bias = np.concatenate((item_bias, ib))
            print(item_vecs.shape, item_bias.shape)"""



    user_vecs = np.genfromtxt(user_vecs_file)
    item_bias = np.genfromtxt(item_bias_file)
    user_bias = np.genfromtxt(user_bias_file)
    global_bias = np.genfromtxt(dir + "global_bias.txt")

    items = Get_items(items_array, n_process, item_set)
    users = Get_items(users_array, n_process, user_set)
    ratings = GetPartOfSparceRatings(users, items)

    best_sgd_model = ExplicitMF(ratings, n_factors=n_factors, learning='sgd', \
                                 item_fact_reg=0.01, user_fact_reg=0.01, \
                                 user_bias_reg=0.01, item_bias_reg=0.01)
    best_sgd_model.item_vecs = item_vecs
    best_sgd_model.item_bias = item_bias
    best_sgd_model.global_bias = global_bias
    best_sgd_model.user_vecs = user_vecs
    best_sgd_model.user_bias = user_bias
    print(Print_result(best_sgd_model, ratings))
    learning_rate = 0.01
    if (user_set > item_set):
        learning_rate *= 1. / sqrt(user_set - item_set + 1.)
    else:
        learning_rate *= 1./ sqrt(user_set - item_set + 1. + n_process)
    best_sgd_model.train(30, learning_rate=learning_rate, from_scratch=False)
    print(Print_result(best_sgd_model,ratings))

    np.savetxt(item_vecs_file, best_sgd_model.item_vecs)
    np.savetxt(item_bias_file, best_sgd_model.item_bias)
    np.savetxt(user_vecs_file, best_sgd_model.user_vecs)
    np.savetxt(item_vecs_file, best_sgd_model.item_vecs)

def CreateDataFirst(n_factors, n_process):
    dir_with_data = data_dir
    ratings = GetData1(dir_with_data)
    n_users_in_test = n_test_users
    train, test, users_order = GetTestUsers(ratings, n_users_in_test, dir)
    save_sparse_matrix(dir + "tes_ratings.txt", test)
    save_sparse_matrix(dir + "train_ratings.txt", train)
    n_users = train.shape[0]
    n_items = train.shape[1]
    users_array = np.arange(n_users)
    np.random.shuffle(users_array)
    items_array = np.arange(n_items)
    np.random.shuffle(items_array)

    np.savetxt(dir + "users_array.txt", users_array)
    np.savetxt(dir + "items_array.txt", items_array)
    item_bias = np.zeros(n_items)
    item_vecs = np.random.normal(scale=1. / n_factors,
                                      size=(n_items, n_factors))
    user_vecs = np.random.normal(scale=1. / n_factors, \
                                      size=(n_users, n_factors))
    user_bias = np.zeros(n_users)

    non_zero = train.nonzero()
    print(non_zero[0], non_zero[1])
    global_bias = 0
    for i in range(non_zero[0].shape[0]):
        global_bias += train[non_zero[0][i], non_zero[1][i]]
    global_bias /= non_zero[0].shape[0]
    print("gloabal bias = " ,global_bias)
    np.savetxt(dir + "items.txt", item_vecs)
    np.savetxt(dir + "items_bias.txt", item_bias)
    np.savetxt(dir + "users_train.txt", user_vecs)
    np.savetxt(dir + "user_bias_train.txt", user_bias)
    with open(dir + "global_bias.txt", 'w') as global_bias_:
        global_bias_.write(str(global_bias))
    item_step = n_items / n_process
    user_step = n_users / n_process
    for i in xrange(n_process):
        np.savetxt(item_dir + "vecs/" + str(i), item_vecs[item_step * i : item_step * (i+1)])
        np.savetxt(item_dir + "bias/" + str(i), item_bias[item_step * i : item_step * (i+1)])
        np.savetxt(user_dir + "vecs/" + str(i), user_vecs[user_step * i : user_step * (i+1)])
        np.savetxt(user_dir + "bias/" + str(i), user_bias[user_step * i : user_step * (i+1)])

def MergeData(n_process):
    items_array = np.genfromtxt(dir + "items_array.txt").astype(int)
    user_array = np.genfromtxt(dir + "users_array.txt").astype(int)
    item_vecs_old = np.genfromtxt(dir + "items.txt")
    item_bias_old = np.genfromtxt(dir + "items_bias.txt")
    user_vecs_old = np.genfromtxt(dir + "users_train.txt")
    user_bias_old = np.genfromtxt(dir + "user_bias_train.txt")
    item_vecs = ''
    item_bias = ''
    user_vecs = ''
    user_bias = ''
    for i in xrange(n_process):
        LIV = np.genfromtxt(item_dir + "vecs/" + str(i))
        LIB = np.genfromtxt(item_dir + "bias/" + str(i))
        LUV = np.genfromtxt(user_dir + "vecs/" + str(i))
        LUB = np.genfromtxt(user_dir + "bias/" + str(i))
        if (i == 0):
            item_vecs = LIV
            item_bias = LIB
            user_vecs = LUV
            user_bias = LUB
        else:
            item_vecs = np.concatenate((item_vecs, LIV))
            item_bias = np.concatenate((item_bias, LIB))
            user_vecs = np.concatenate((user_vecs, LUV))
            user_bias = np.concatenate((user_bias, LUB))

    item_sort = np.argsort(items_array[:item_vecs.shape[0]])
    user_sort = np.argsort(user_array[:user_vecs.shape[0]])
    #print(items_array[item_sort])

    """item_vecs_old[items_array[item_sort]] = 0
    item_vecs_old[items_array[item_sort]] = item_vecs[item_sort]
    item_bias_old[items_array[item_sort]] = item_bias[item_sort]
    user_vecs_old[user_array[user_sort]] = user_vecs[user_sort]
    user_bias_old[user_array[user_sort]] = user_bias[user_sort]"""

    item_vecs_old[items_array[:item_vecs.shape[0]]] = item_vecs
    item_bias_old[items_array[:item_vecs.shape[0]]] = item_bias
    user_vecs_old[user_array[:user_vecs.shape[0]]] = user_vecs
    user_bias_old[user_array[:user_vecs.shape[0]]] = user_bias

    np.savetxt(dir + "items1.txt", item_vecs_old)
    np.savetxt(dir + "items_bias1.txt", item_bias_old)
    np.savetxt(dir + "users_train1.txt", user_vecs_old)
    np.savetxt(dir + "user_bias_train1.txt", user_bias_old)

def MakeNormalRatingFormat(rating_old_file, rating_new_file):
    test_ratings = np.load(rating_old_file)
    test_rat = np.zeros(test_ratings['shape'])
    data = test_ratings['data']
    col = test_ratings['col']
    rows = test_ratings['row']
    item_popularity = np.zeros(test_ratings['shape'][1])
    for i in range(len(rows)):
        test_rat[rows[i], col[i]] = data[i]
        item_popularity[col[i]] -= 1
    sorted_item_popularity = np.argsort(item_popularity)
    print(item_popularity[sorted_item_popularity[1000]], test_rat.shape[0])
    np.savetxt(rating_new_file, test_rat)
    np.savetxt(rating_new_file + "_", sorted_item_popularity[:1000])
    
    return test_rat

def TrainTestUsers():
    #test_rat = MakeNormalRatingFormat(dir + "tes_ratings.txt.npz", dir + "tes_ratings.txt")
    test_rat = np.genfromtxt(dir + "tes_ratings.txt")
    popular_items = np.genfromtxt(dir + "train_ratings.txt_").astype(int)
    #train = test_rat
    #train, test = train_test_split(test_rat)
    train = np.zeros(test_rat.shape)
    train = train.T
    train[popular_items] = test_rat.T[popular_items]
    train = train.T
    test_rat = test_rat.T
    print(len(test_rat.nonzero()[0]))
    test_rat[popular_items] = 0.
    print(len(test_rat.nonzero()[0]))
    test_rat = test_rat.T
    model = ExplicitMF(train, n_factors=n_factors, learning='sgd', \
                                 item_fact_reg=0.01, user_fact_reg=0.01, \
                                 user_bias_reg=0.01, item_bias_reg=0.01)
    model.item_vecs = np.genfromtxt(dir + "items1.txt")
    model.item_bias = np.genfromtxt(dir + "items_bias1.txt")
    model.global_bias = np.genfromtxt(dir + "global_bias.txt")
    model.user_vecs = np.genfromtxt(dir + "users_train1.txt")
    model.user_bias = np.genfromtxt(dir + "user_bias_train1.txt")
    model.user_bias = np.mean(train, axis=1)
    for i in range(train.shape[0]):
        non_zero = train[i].nonzero()[0]
        questions = model.item_vecs[non_zero]
        print(questions.shape)
        answers = train[i][non_zero] - model.item_bias[non_zero] - model.user_bias[i] - model.global_bias
        new_inverse_matrix = np.linalg.inv(np.dot(questions.T, questions) + 0.01 * np.eye(questions.shape[1]))
        model.user_vecs[i] = np.dot(new_inverse_matrix,
                         np.dot(np.array(answers), np.array(questions)))
        
    #model.train(10, learning_rate=0.01, from_scratch=False, user_step=True, item_step=False)
    print(Print_result(model, test_rat))
    np.savetxt(dir + "user_bias.txt", model.user_bias)
    np.savetxt(dir + "users.txt", model.user_vecs)

"""import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--i', type = int)
parser.add_argument('--u', type = int)
FLAGS, unparsed = parser.parse_known_args()
"""
if (FLAGS.i == 0):
    CreateDataFirst(n_factors, n_process)
if (FLAGS.i == 1):
    from multiprocessing import Process
    for j in range(n_process):
        threads = []
        for i in range(n_process):
            threads.append(Process(target=RunOneProcess, args=((i + j)%n_process, i, n_process)))
        for i in range(n_process):
             threads[i].start()
        for i in range(n_process):
            threads[i].join()
if (FLAGS.i == 2):
    MergeData(n_process)
if (FLAGS.i == 3):
    TrainTestUsers()
if (FLAGS.i == 4):
    MakeNormalRatingFormat(dir + "tes_ratings.txt.npz", dir + "tes_ratings.txt")
    MakeNormalRatingFormat(dir + "train_ratings.txt.npz", dir + "train_ratings.txt")
