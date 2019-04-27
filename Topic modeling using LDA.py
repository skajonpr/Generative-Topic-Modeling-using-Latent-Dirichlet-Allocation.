import pandas as pd
from sklearn import metrics
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from numpy.random import shuffle
import json
import numpy as np

# define a funtion named "cluster_lda" to do topic modeling.
def cluster_lda(train_file, test_file):
    
    # Import train dataset from json file.
    train_data=json.load(open(train_file,'r'))
    
    # shuffle train data
    shuffle(train_data)
    
    # Import test dataset from json file.
    test_data=json.load(open(test_file,'r'))
    
    # seperate texts and labels into different variables.
    text,label=zip(*test_data)
    test_text=list(text)
    test_label=list(label)
    
    # Select only first index as projecting only one cluster per document.
    # However, in the given json file, each document label could have multi cluster.
    test_label = [ i[0] for i in test_label]
    
    # Define verterizor with parameters.
    tf_vectorizer = CountVectorizer(max_df=0.999, \
                min_df=5, stop_words='english')

    # vectorize of training dataset. 
    tf = tf_vectorizer.fit_transform(train_data)
    
    # define number of clusters.
    num_topics = 3
    
    # Fit model using LDA
    lda = LatentDirichletAllocation(n_components=num_topics, \
                                    max_iter=25,verbose=1,
                                    evaluate_every=1, n_jobs=1,
                                    random_state=0).fit(tf)
    
    # create vectorize test data.
    test_text_tf = tf_vectorizer.transform(test_text)
    
    # Predict clusters of test data by computing proabilities of each cluster
    # of each news.
    topic_assign = lda.transform(test_text_tf)
    
    # copy predicted data
    topics = np.copy(topic_assign)
    
    lda_topics = []
    cluster = []
    
    # find the index of highest proability of each clusers and store its index in a list.
    for probs in topics:
        cluster.append(probs.argsort()[::-1][0])

    # Define a dataframe of test labels and predicted clusters.
    confusion_df = pd.DataFrame(list(zip( test_label, cluster)),\
                            columns = ["label", "cluster"])
                            
    # print out the crosstab.
    print (pd.crosstab( index=confusion_df.cluster, columns=confusion_df.label))             
     
                
    cluster_dict = {}
    # print Crosstab to see the correctly predicted number of each cluster/ 
    get_col = pd.crosstab( index=confusion_df.cluster, columns=confusion_df.label).columns.values
    
    # store values in array from created crosstab
    get_cluster_dict  = pd.crosstab( index=confusion_df.cluster, columns=confusion_df.label).values
    
    # get dictionary of cluster number and cluster name (eg. {0 : 'Travel & Transportation'....}) 
    for idx , arr in enumerate(get_cluster_dict) :
        cluster_dict[idx] = get_col[arr.argsort()[::-1][0]]
        print ('Cluster {} : Topic {}'.format(idx, cluster_dict[idx]))
    
    # get a list of predicted cluster
    for probs in topics:    
        lda_topics.append(cluster_dict[probs.argsort()[::-1][0]])

    # report performance   
    print(metrics.classification_report(test_label, lda_topics))
    
    return topics, list(label)