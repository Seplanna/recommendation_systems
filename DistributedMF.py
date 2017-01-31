from RunPMF import GetTestUsers
from RunPMF import GetData1
from RunPMF import save_sparse_matrix
from PMF import *
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--i', type = int)
parser.add_argument('--u', type = int)
parser.add_argument('--f', type = int)
FLAGS, unparsed = parser.parse_known_args()

data_dir = "../DATA/ml-20m"
dir = "../GeneratedData/data" + str(FLAGS.f) + "/"
item_dir = "../GeneratedData/data" + str(FLAGS.f) + "/ITEMS/"
user_dir = "../GeneratedData/data" + str(FLAGS.f) + "/USERS/"

n_factors = 80
def GetPartOFRating(users, items):
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

def Get_items(items_array, n_process, process):
    item_step = int(len(items_array) / n_process)
    items_ = items_array[item_step * process: item_step * (process+1)]
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

def RunOneProcess(user_set, item_set, n_process):
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
    user_vecs = np.genfromtxt(user_vecs_file)
    item_bias = np.genfromtxt(item_bias_file)
    user_bias = np.genfromtxt(user_bias_file)
    global_bias = np.genfromtxt(dir + "global_bias.txt")

    items = Get_items(items_array, n_process, item_set)
    users = Get_items(users_array, n_process, user_set)
    ratings = GetPartOFRating(users, items)
    best_sgd_model = ExplicitMF(ratings, n_factors=n_factors, learning='sgd', \
                                 item_fact_reg=0.01, user_fact_reg=0.01, \
                                 user_bias_reg=0.01, item_bias_reg=0.01)
    best_sgd_model.item_vecs = item_vecs
    best_sgd_model.item_bias = item_bias
    best_sgd_model.global_bias = global_bias
    best_sgd_model.user_vecs = user_vecs
    best_sgd_model.user_bias = user_bias
    best_sgd_model.train(50, learning_rate=0.01, from_scratch=False)

    np.savetxt(item_vecs_file, best_sgd_model.item_vecs)
    np.savetxt(item_bias_file, best_sgd_model.item_bias)
    np.savetxt(user_vecs_file, best_sgd_model.user_vecs)
    np.savetxt(item_vecs_file, best_sgd_model.item_vecs)

def CreateDataFirst(n_factors, n_process):
    dir_with_data = "../DATA/ml-20m"
    ratings = GetData1(dir_with_data)
    n_users_in_test = 1000
    train, test, users_order = GetTestUsers(ratings, n_users_in_test, dir)
    np.savetxt(dir + "users_order", users_order)
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

def MergeData(n_factors, n_process):
    items_array = np.genfromtxt(dir + "items_array.txt")
    item_vecs = ''
    item_bias = ''
    user_vecs = ''
    user_bias = ''
    for i in xrange(n_process):
        lIV = np.genfromtxt(item_dir + "vecs/" + str(i))
        np.genfromtxt(item_dir + "bias/" + str(i))
        np.genfromtxt(user_dir + "vecs/" + str(i))
        np.genfromtxt(user_dir + "bias/" + str(i))
if (FLAGS.i == 0):
    CreateDataFirst(n_factors, 10)
"""import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--i', type = int)
parser.add_argument('--u', type = int)
FLAGS, unparsed = parser.parse_known_args()
"""
if (FLAGS.i == 1):
    from multiprocessing import Process
    for j in range(10):
        threads = []
        for i in range(10):         
            threads.append(Process(target=RunOneProcess, args=((i + j)%10, i, 10)))
        for i in range(10):
            threads[i].start()
        for i in range(10):
            threads[i].join()
