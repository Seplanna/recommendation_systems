import sys
import gzip
import numpy as np

data_dir = '../DATA/Amazone'

def GetRatingsFromReview(path, output):
  g = open(path, 'r')
  out = open(output, 'w')
  line_n = 0
  for l in g:
    sys.stdout.write("\r%d%%" % line_n)
    line_n += 1 
    item_data = eval(l)
    out.write(str(item_data['reviewerID']) + "," + str(item_data['asin']) + "," + str(item_data['overall']) + '\n')

def GetMostPopularItems(path, output):
    g = open(path)
    out = open(output, 'w')
    n_ratings = []
    all_ratings = []
    last_item = -1
    line_n = 0
    for l in g:
        sys.stdout.write("\r%d%%" % line_n)
        line_n += 1
        item = l.strip().split(',')[1]
        if (item != last_item):
           all_ratings.append([])
           last_item = item
           n_ratings.append(0)
        all_ratings[-1].append(l)
        n_ratings[-1] += 1
        
    n_ratings = np.array(n_ratings)
    sort_ratings = np.argsort(-n_ratings)
    for i in range(30000):
        out.write("".join(all_ratings[sort_ratings[i]]))

def GetItemPopularity(path, output):
    items_popularity = {}
    with open(path) as g:
        for l in g:
            l = l.strip().split(',')
            item = l[1]
            if (item not in items_popularity):
                items_popularity[item] = 0
            items_popularity[item] += 1
    items_popularity_list = [[p,items_popularity[p]] for p in items_popularity.keys()]
    items_popularity_list.sort(key=lambda x:-x[1])
    a = [str(items_popularity_list[i][0]) + "," + str(items_popularity_list[i][1]) for i in xrange(len(items_popularity_list))]
    with open(output, 'w') as r:
        r.write("\n".join(a))

def GetMetaPopularItems(meta_path, path, output):
    g = open(path, 'r')
    m = open(meta_path, 'r')
    out = open(output, 'w')
    items = set()
    for l in g:
        items.add(l.split(',')[1])
    for l in m:
        f = eval(l)
        if (f['asin'] in items):
            out.write(l)


def ItemsToNumbers(path, output, output_items):
    g = open(path, 'r')
    out = open(output, 'w')
    items = open(output_items, 'w')
    item_n = -1
    previous_item = -1
    for l in g:
        l = l.strip().split(',')
        current_item = l[1]
        if previous_item != current_item:
            item_n += 1
            previous_item = current_item
            items.write(str(l[1]) + "," + str(item_n) + '\n')
        out.write(str(l[0]) + "," + str(item_n) + ',' + str(l[2]) + '\n')

def SortLines(path, output):
    g = open(path, 'r')
    out = open(output, 'w')
    l = g.readlines()
    l.sort()
    out.write(''.join(l))

def GetMostPopularUsers(path, output):
    g = open(path)
    out = open(output, 'w')
    n_ratings = []
    all_ratings = []
    last_user = -1
    line_n = 0
    for l in g:
        sys.stdout.write("\r%d%%" % line_n)
        line_n += 1
        user = l.strip().split(',')[0]
        if (user != last_user):
           all_ratings.append([])
           last_user = user
           n_ratings.append(0)
        all_ratings[-1].append(l)
        n_ratings[-1] += 1

    n_ratings = np.array(n_ratings)
    sort_ratings = np.argsort(-n_ratings)
    for i in range(200000):
        out.write("".join(all_ratings[sort_ratings[i]]))

def UsersToNumbers(path, output, output_items):
    g = open(path, 'r')
    out = open(output, 'w')
    items = open(output_items, 'w')
    item_n = -1
    previous_item = -1
    to_write = []
    user_bias = 0.
    for l in g:
        l = l.strip().split(',')
        current_item = l[0]
        if previous_item != current_item:
            if (len(to_write) > 0):
                user_bias /= len(to_write)
                for r in to_write:
                    out.write(str(r[0]) + "," + str(r[1]) + "," + str(r[2]) + "\n")
            item_n += 1
            previous_item = current_item
            user_bias = 0.
            to_write = []
            items.write(str(l[0]) + "," + str(item_n) + '\n')
        user_bias += float(l[2])
        to_write.append([item_n, str(l[1]), float(l[2])])

def GetNegativeSampling(path, output, item_pop_):
    g = open(path, 'r')
    out = open(output, 'w')
    min_n_of_ratings = 20
    items = {}
    items_per_user = []
    last_user = -1
    line_n = 0
    with open(path) as g: 
        for l in g:
            line_n += 1 
            item = l.strip().split(',')[1]
            user = l.strip().split(',')[0]
            if (user != last_user):
               if (len(items_per_user) < 1000 and len(items_per_user) >= min_n_of_ratings):
                   for item_ in items_per_user:
                       if (item_ not in items):
                           items[item_] = 0
                       items[item_] += 1
               if(last_user != -1 and len(items_per_user) < min_n_of_ratings):
                   break
               items_per_user = []
               last_user = user
            items_per_user.append(item)
    item_popularity = [[i, items[i]] for i in items]
    item_popularity.sort(key=lambda x:-x[1])
    popular_items = set([item_popularity[i][0] for i in range(1000)])
    test_items = set([item_popularity[i][0] for i in range(1000, len(item_popularity))])
    item_pop = open(item_pop_, 'w')
    item_pop.write('\n'.join(str(i[0]) + "," + str(i[1]) for i in item_popularity))

    last_user = -1
    user_items = []
    n_ratings_per_user = 0
    to_w_rite = []
    line_n = 0
    with open(path) as g:
        for l in g:
            line_n += 1
            user = l.strip().split(',')[0]
            if (line_n%10000 == 0):
                print(line_n)
                #sys.stdout.write("\r%d%%" % line_n)
 
            if (user != last_user):
               if(last_user != -1 and n_ratings_per_user < 1000):
                  if (n_ratings_per_user < min_n_of_ratings):
                      print("AAAA")
                      return
                  candidate_items = popular_items.difference(set(user_items))
                  candidate_items_ = [i for i in candidate_items]
                  
                  negative_samples = np.random.choice(candidate_items_, len(popular_items) - len(candidate_items), False)
                  out.write(''.join(to_write))
                  for i in negative_samples:
                      out.write(str(last_user) + "," + str(i) + ",-1\n")
                  
                  candidate_items = test_items.difference(set(user_items))
                  candidate_items_ = [i for i in candidate_items]
                  
                  negative_samples = np.random.choice(candidate_items_, len(test_items) - len(candidate_items), False)
                  for i in negative_samples:
                      out.write(str(last_user) + "," + str(i) + ",-1\n")
               last_user = user
               user_items = []
               to_write = []
               n_ratings_per_user = 0
            user_items.append(l.strip().split(',')[1])
            n_ratings_per_user += 1
            to_write.append(l)      
       


def GetItemRating(path, output):
    g = open(path, 'r')
    out = open(output, 'w')
    result = {}
    for l in g:
        l = l.strip().split(',')
        if (l[1] not in result):
            result[l[1]] = [0., 0.]
        if (float(l[2]) > 0):
           result[l[1]][0] += 1.
        else:
           result[l[1]][1] += 1.
    a = [[r, result[r][0] / (result[r][0]+ result[r][1])] for r in result]
    a.sort(key=lambda x:x[1])
    out.write('\n'.join(str(i[0]) + "," + str(result[i[0]][0]) + "," + str(result[i[0]][1])  for i in a))
    
def GetMeanRating(path):
    g = open(path, 'r')
    n_lines = 0.
    summ = 0. 
    for l in g:
        l = l.strip().split(',')
        summ += float(l[2])
        n_lines += 1.
    print(summ / n_lines)
        
def PrepareData():
    GetRatingsFromReview(data_dir + "/Books_5.json", data_dir + "/test_ratings.csv")
    GetMostPopularItems(data_dir + "/test_ratings.csv", data_dir + "/popular_items.csv")
    GetMetaPopularItems(data_dir + "/meta_Books.json", data_dir + "/popular_items.csv", data_dir + "/meta_Books_popularItems.json")
    ItemsToNumbers(data_dir + "/popular_items.csv", data_dir + "/item_ratings.csv", data_dir + "/items")
    SortLines(data_dir + "/item_ratings.csv", data_dir + "/sort_item_ratings.csv")
    GetMostPopularUsers(data_dir + "/sort_item_ratings.csv", data_dir + "/sort_item_ratings_popular_users.csv")
    UsersToNumbers(data_dir + "/sort_item_ratings_popular_users.csv", data_dir + "/ratings.csv", data_dir + "/users")
    GetNegativeSampling(data_dir + "/ratings.csv", data_dir + "/ratings_negative1.csv", data_dir + "/item_popularity")
    #GetItemRating(data_dir + "/ratings_negative1.csv", data_dir + "/item_popularity")
#PrepareData()
GetItemPopularity(data_dir + "/ratings.csv", data_dir + "/item_popularity_final")
