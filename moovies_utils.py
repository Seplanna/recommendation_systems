def GetMoviesName(movie_file):
    movies= {}
    with open(movie_file) as f:
        line_n = -1
        for l in f:
            l = l.strip().split(',')
            movies[l[0]] = l[1]
    return movies

def ItemPopularity(ratings_file, out):
    movies = GetMoviesName('../DATA/ml-20m/movies.csv')
    popularity = {}
    with open(ratings_file) as f:
        line_n = -1 
        for l in f:
            l = l.strip().split(',')
            line_n += 1
            if (line_n == 0):
                continue
            if l[1] not in popularity:
               popularity[l[1]] = [0, 0]
            popularity[l[1]][0] += 1
            popularity[l[1]][1] += (float(float(l[2]) > 3.5) - 0.5) * 2
    popularity_list = [[k, popularity[k][0], popularity[k][1] / popularity[k][0]] for k in popularity.keys()]
    popularity_list.sort(key=lambda x: x[1])
    print(len(popularity_list))
    with open(out, 'w') as f:
        f.write('\n'.join(movies[s[0]] + ',' + str(s[0]) + ',' + str(s[1]) + ',' + str(s[2]) for s in popularity_list))

def FilterMovies(popularity_file, rating_file, new_rating_file):
    moovies = set()
    with open(popularity_file) as p:
        line_n = -1
        for l in p:
            l = l.strip().split(',')
            if (int(l[2]) < 5 or abs(float(l[3])) > 0.5):
               continue
            line_n += 1
            moovies.add(l[1])
    with open(rating_file) as f, open(new_rating_file, 'w') as r:
       for l in f:
           if l.split(',')[1] in moovies:
               r.write(l)

def GetTranslationMovies(popularity_file):
    movies = {}
    with open(popularity_file) as m:
        line_n = 0
        for l in m:
            l = l.strip().split(',')
            movies[l[1]] = line_n
            line_n += 1
    return movies

 
def GangeRatingsMoviesNotSparce(popularity_file, old_ratings, new_ratings):
    movies = GetTranslationMovies(popularity_file)
    with open(old_ratings) as o_r, open(new_ratings, 'w') as n_r:
      for l in o_r:
          l = l.strip().split(',')
          l[1] = movies[l[1]]
          n_r.write(','.join(str(s) for s in l) + '\n')  

def GetUsersWithoutBias(old_ratings, new_ratings):
    line_n = 0
    previous_user = -1
    user_bias = 0.
    to_write = []
    with open(old_ratings) as o_r, open(new_ratings, 'w') as n_r:
        for l in o_r:
            l = l.strip().split(',')
            user = l[0]
            if (user != previous_user):
                if (len(to_write) > 5):
                    bias = user_bias / len(to_write)
                    if (bias > 0.3 and bias < 0.7):
                        line_n += 1
                        n_r.write('\n'.join(to_write))
                previous_user = user
                user_bias = 0
                to_write = []
            to_write.append(str(line_n) + "," + ','.join(str(i) for i in l[1:]))
            user_bias += float(float(l[2]) > 3.5)



#ItemPopularity("../DATA/ml-20m/ratings.csv", '../DATA/ml-20m/popularity.txt')     
FilterMovies('../DATA/ml-20m/popularity.txt', "../DATA/ml-20m/ratings.csv", '../DATA/ml-20m/ratings1.csv')     
ItemPopularity("../DATA/ml-20m/ratings1.csv", '../DATA/ml-20m/popularity1.txt')     
GangeRatingsMoviesNotSparce("../DATA/ml-20m/popularity1.txt", "../DATA/ml-20m/ratings1.csv", "../DATA/ml-20m/ratings2.csv")
GetUsersWithoutBias("../DATA/ml-20m/ratings2.csv",   "../DATA/ml-20m/ratings3.csv")
