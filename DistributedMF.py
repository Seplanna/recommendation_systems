from RunPMF import GetTestUsers
from RunPMF import GetData1
from RunPMF import save_sparse_matrix
from PMF import *
import pandas as pd

dir = '../../PWL/dist/'
item_dir = '../../PWL/dist/ITEMS/'
user_dir = "../../PWL/dist/USERS/"

def GetPartOFRating(users, items, dir):
    df = pd.read_csv(dir + "/" + 'ratings.csv')
    n_users = users.shape[0]
    n_items = items.shape[0]
    print(n_users, n_items)
    ratings = np.zeros((n_users, n_items))
    #ratings = lil_matrix((n_users, n_items))
    print(ratings.shape)
    for row in df.itertuples():
        if row[1] - 1 in users and row[2] - 1 in items:
            ratings[users[row[1] - 1], items[row[2] - 1]] = (int(row[3] > 3.5) - 0.5) * 2
    return ratings

def Get_items(items_array, n_process, process):
    item_step = items_array / n_process
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

def RunOneProcess(user_set, item_set, n_process, dir, n_factors = 100):

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
    global_bias = ""

    items = Get_items(items_array, n_process, item_set)
    users = Get_items(users_array, n_process, user_set)
    ratings = GetPartOFRating(users, items, dir)
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
    dir_with_data = "../../dataset/ml-20m"
    ratings = GetData1(dir_with_data)
    n_users_in_test = 1000
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


    global_bias = np.mean(ratings[np.where(ratings != 0)])
    np.savetxt(dir + "items.txt", item_vecs)
    np.savetxt(dir + "items_bias.txt", item_bias)
    np.savetxt(dir + "users_train.txt", user_vecs)
    np.savetxt(dir + "user_bias_train.txt", user_bias)
    with open(dir + "global_bias.txt", 'w') as global_bias:
        global_bias.write(str(global_bias))
    item_step = n_items / n_process
    user_step = n_users / n_process
    for i in xrange(n_process):
        np.savetxt(item_dir + "vecs/" + str(i), item_vecs[item_step * i : item_step * (i+1)])
        np.savetxt(item_dir + "bias/" + str(i), item_bias[item_step * i : item_step * (i+1)])
        np.savetxt(user_dir + "vecs/" + str(i), user_vecs[user_step * i : user_step * (i+1)])
        np.savetxt(user_dir + "bias/" + str(i), user_bias[user_step * i : user_step * (i+1)])

def MergeData(n_process):
    items_array = np.genfromtxt(dir + "items_array.txt")
    user_array = np.genfromtxt(dir + "users_array.txt")
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
    item_sort = np.argsort(items_array)
    user_sort = np.argsort(user_array)
    #print(items_array[item_sort])
    item_vecs_old[items_array[item_sort]] = item_vecs[item_sort]
    item_bias_old[items_array[item_sort]] = item_bias[item_sort]
    user_vecs_old[user_array[user_sort]] = user_vecs[user_sort]
    user_bias_old[user_array[user_sort]] = user_bias[user_sort]
    np.savetxt(dir + "items1.txt", item_vecs_old)
    np.savetxt(dir + "items_bias1.txt", item_bias_old)
    np.savetxt(dir + "users_train1.txt", user_vecs_old)
    np.savetxt(dir + "user_bias_train1.txt", user_bias_old)

#CreateDataFirst(100, 10)
MergeData(10)