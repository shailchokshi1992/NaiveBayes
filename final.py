from pprint import pprint
import numpy as np
from math import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import logging



class NaiveBayesTextClassifier():
    def __init__(self):
        self.feat_prob= [] #dict to store feature probabilities
        self.class_prob= [] #dict to store class probabilities
        self.num_class= [] #dict to store class number
        self.num_feat= [] # dict to store feature number
        self.alpha = 0.05

    def train(self,X,y):
        each_class_count = {} # list for counting each class
        feature_count= {} # list for counting each feature
        #alpha = 1
        temp = [] # temp dict to store
        temp.append(np.unique(y))
        self.num_class.append(temp[0].size) #total number of classes
        self.num_feat.append(X[0].size)#total number of features
        #print y.size
        #print X.size
        # print ("\n")
        # print self.num_class
        # print self.num_feat
        for i in range(y.size):
            if y[i] in feature_count:
                continue
            else:
                feature_count[y[i]] = [0 for w in range (X[i].size)]

        #count the features per each class across train, count occurance of each class across train
        for i in range (y.size):
            if y[i] in each_class_count:
                each_class_count[y[i]] +=1
            else:
                each_class_count[y[i]] = 1

            for j in  range(X[i].size):
                    feature_count[y[i]][j] += X[i][j]

        # print feature_count[0]

        # Calculate class and feature probablities per each class
        for cls in feature_count:
            num = (self.alpha+each_class_count[cls])
            din = (y.size+(self.num_class[0]*self.alpha))
            self.class_prob.append((num/float(din)))
            temp_ar = np.array([])
            for j in  range(X[i].size):
                num= (feature_count[cls][j] + self.alpha)
                din = (each_class_count[cls]+(2*self.alpha))
                temp_ar=np.append(temp_ar,(num/float(din)))
            self.feat_prob.append(temp_ar)


        print self.class_prob[0]
        print("\n")
        print len(self.feat_prob[0])

    def predict(self, X):
        print ("Predicting Naive bayes....!!!!")
        Y_predict = np.array([])
        for i in X:
            neg_log_prob = 0
            minimum_neg_log_prob=999999999999999
            category = 0

            for cls in range(self.num_class[0]):
                neg_log_prob = -log(self.class_prob[cls])
                for j in  range(self.num_feat[0]):
                    if (i[j])==0:
                        neg_log_prob -= log(1-self.feat_prob[cls][j])
                    else:
                        neg_log_prob -= log(self.feat_prob[cls][j])

                if minimum_neg_log_prob>neg_log_prob:
                    category=cls
                    minimum_neg_log_prob=neg_log_prob

            Y_predict=np.append(Y_predict,category)
        return Y_predict

logging.basicConfig()
print("DATA LOADING......!!!")
### Load 20 news group database with subset train from http://www.cs.cmu.edu/afs/cs/project/theo-11/www/naive-bayes/20_newsgroups.tar.gz
remove_words = ('headers', 'footers', 'quotes')  # Clean data by removing quotes,headers and footers
newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True, remove=remove_words)
newsgroups_test = fetch_20newsgroups(subset='test',shuffle=True, remove=remove_words)
print ("DATA LOADED......!!!")

y_train, y_test = newsgroups_train.target, newsgroups_test.target
#print y_train
#pprint(list(newsgroups_train.target_names))
#print (newsgroups_train.filenames)
#sprint (newsgroups_train.target)
no_features = 1000
vectorizer = TfidfVectorizer(stop_words='english', binary=True, max_features = no_features)#, analyzer='char', ngram_range=(1,3))
X_train = vectorizer.fit_transform(newsgroups_train.data).toarray()
#print (X_train.shape)
X_test = vectorizer.transform(newsgroups_test.data).toarray()
#print (X_train.test)
feature_names = vectorizer.get_feature_names() # here list of different words used.
#print feature_names

classsifier = NaiveBayesTextClassifier()
classsifier.train(X_train,y_train)
y_pred = classsifier.predict(X_test)
print type(y_pred)
print type(y_test)
print ("accuracy: %f"%(np.mean((y_test-y_pred)==0)))
