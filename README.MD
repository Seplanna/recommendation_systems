Probabilistic matrix factorization (PMF) in Python  

Parameters:   
>num_feat: Number of latent features,  
>epsilon: learning rate,  
>_lambda: L2 regularization,  
>momentum: momentum of the gradient,  
>maxepoch: Number of epoch before stop,  
>num_batches: Number of batches in each epoch (for SGD optimization),  
>batch_size: Number of training samples used in each batches (for SGD optimization)  

Methods:  
>fit(train_tuple, val_tuple)  
>>Fit the model with train_tuple and evaluate RMSE on both train and validation data.  
>>Input tuple format:  (userID, movieID, rating)  # 0-index ID is recommended  
>>Output: U and V matrices, RMSE Error on Train and Validation after each epoch.  

>predict(userID)  
>>Predict rating of all movies for the given user.  

>set_params(parameter_dict)  
>>Set parameters by providing a parameter dictionary.  

>>Helper function:  
>>>def wrap_Parameters(num_feat, epsilon, _lambda, momentum, maxepoch, num_batches, batch_size):  
>>>>return {"num_feat": num_feat, "epsilon":epsilon, "_lambda":_lambda, "momentum":momentum, "maxepoch":maxepoch,   "num_batches":num_batches, "batch_size":batch_size}  


Reference:  
1. Mnih, A., & Salakhutdinov, R. (2007). Probabilistic matrix factorization. In Advances in neural information processing systems (pp. 1257-1264).  
2. Salakhutdinov, R. Probabilistic matrix factorization in Matlab. http://www.cs.toronto.edu/~rsalakhu/code_BPMF/pmf.m.

DistributedPMF

Reference:
https://code.facebook.com/posts/861999383875667/recommending-items-to-more-than-a-billion-people/

is a distributed version of PMF.
parameters: n_process --- number of sets of items
            data_dir --- directory with ratings
            dir --- directory with result vectors of users and items. Should contain ITEMS and USERS directories 
            item_dir --- directory for items' vectors. It needs for training
            user_dir --- directory for users' vectors. It needs for training. 
            n_test_users --- number users in the test, cold users 
            n_factors --- dimension of latent space

to run taining : 1.first use parameter --i 0. This split items and users on the subsets
                 2. --i 1. This needs to run training  
                 3. --i 2. This needs to 
                 4. --i 4
                 5. --i 3
