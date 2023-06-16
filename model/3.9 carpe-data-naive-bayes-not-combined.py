import pandas as pd
import nltk
import numpy as np
from numpy import random as npr
import time
import string
import sklearn
#from itertools import chain, imap
#nltk.download('punkt')
#nltk.download("stopwords")
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import re
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
#from operator import itemgetter
df1 = pd.read_json('C:/Users/tommy/Downloads/final_50k_221118.json', lines=True)
#Read in data
df_shuffled = df1.sample(frac=1, random_state=42)

# Reset the index
df_shuffled = df_shuffled.reset_index(drop=True)

first10 = df_shuffled[0:50000]
#Select first 50000 rows
df2 = first10[["business_name","content"]]

label = ["is_entertainment","is_traffic"]
labeldf = first10[["is_entertainment","is_traffic"]]
#Extract business name and content
begin = time.time()
#print(df2)
token = []
name = ["business_name","content"]
dic1 = {}
WORD = re.compile(r'\w+')
def regTokenize(text):
    words = WORD.findall(text)
    return words
for i in [0,1]:
    arrx = []
    for j in range(0,len(df2[name[i]])):
        listx = regTokenize(df2[name[i]][j])
        arrx.append(listx)
    dic1[name[i]] = arrx
#print(dic1)
#dic1 contains all the tokens of business name and contents
end = time.time()
print("tokenization",end-begin)
now = end
stop_words = set(stopwords.words("english"))
for i in range(len(string.punctuation)):    
    stop_words.add(string.punctuation[i])
stop_words.add("'s")
stop_words.add(".....")
#print(stop_words)
stemmer = PorterStemmer()
#print(dic1)
for i in [0,1]:
    arrc = []
    for j in range(0,len(dic1[name[i]])):
        flist = []
        for z in dic1[name[i]][j]:
            if z.casefold() not in stop_words:
                flist.append(stemmer.stem(z))
        arrc.append(" ".join(flist))
    dic1[name[i]] = arrc
    #print(len(dic1[name[i]]))
#print(dic1)
end = time.time()
print("data cleaning",end-now)
now = end
tflist = []
for i in [0,1]:
    tf_idf_model  = TfidfVectorizer(min_df=0.001)
    #print(tf_idf_model.get_feature_names_out())
    corpus = dic1[name[i]]
    tf_idf_vector = tf_idf_model.fit_transform(corpus)
    tf_idf_array = tf_idf_vector.toarray()
    words_set = tf_idf_model.get_feature_names_out()
    #print(words_set)
    df_tf_idf= pd.DataFrame(tf_idf_array, columns = words_set)
    tflist.append(df_tf_idf)
    #This is the tf-idf result that we can use for SVM or Simple Bayes
    #print(df_tf_idf)
    #print(labeldf[label[i]])
    '''
    #tfidf = []
    tfidf = [(j,sum(df_tf_idf[j])) for j in words_set]
    tfidf.sort(key=lambda x: x[1],reverse=True)
    print(name[i],tfidf)
    end = time.time()
    print(name[i],end-now)
    now = end'''
npr.seed(123)
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
# perform encoding on training and testing dataset to differentiate between different labels and assign them to 0 or 1
for i in [0]:
    for j in [0,1]:
        print(name[i],label[j])
        X_train, X_test, y_train, y_test = train_test_split(tflist[i], labeldf[label[j]], test_size=0.2, random_state=42)
        print(X_train.shape,y_train.shape)
        # create and fit the Naive Bayes model
        nb_model = MultinomialNB()
        #nb_model = GaussianNB()
        nb_model.fit(X_train, y_train)

        # make predictions on the test data
        y_pred = nb_model.predict(X_test)
        '''Encoder = LabelEncoder()
        y_train_svm = Encoder.fit_transform(y_train)
        y_test_svm = Encoder.fit_transform(y_test)
        SVM = svm.SVC(C=1.0, kernel = 'linear', degree = 3, gamma = 'auto')
        SVM.fit(X_train,y_train_svm)
        predictions_SVM = SVM.predict(X_test)
        print("Accuracy: ", accuracy_score(predictions_SVM, y_test_svm)*100)'''
        
        '''
        param_grid_nb = {
        'var_smoothing': np.logspace(0,-9, num=100)
        }
        nbModel_grid = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid_nb, verbose=1, cv=10, n_jobs=-1)
        nbModel_grid.fit(X_train, y_train)
        print(nbModel_grid.best_estimator_)
        y_pred = nbModel_grid.predict(X_test)'''
        # calculate accuracy
        #accuracy = (y_pred == y_test).sum() / len(y_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')

        
        print(confusion_matrix(y_test, y_pred), ": is the confusion matrix")
       

        print(precision_score(y_test, y_pred), ": is the precision score")

        print(recall_score(y_test, y_pred), ": is the recall score")

        print(f1_score(y_test, y_pred), ": is the f1 score")

        # Get the misclassified examples
        misclassified = X_test[y_pred != y_test]
        mis = misclassified.index.tolist()
        print(mis)
        # Print a few examples of misclassified comments
        '''for x in range(5):
            if x < len(mis): # Check if the index is within bounds
                predictor = df2[name[i]].iloc[mis[x]]#[name[i]] # Replace 'comment' with the actual column name
                actual_label = labeldf[label[i]].iloc[mis[x]]#[label[j]] # Replace 'label' with the actual column name
                print(f"Predictor: {predictor}")
                print(f"Actual label: {actual_label}")
                print("\n")
            else:
                print("No more misclassified examples.")
                break'''
        for x in mis:
            predictor = df2[name[i]].iloc[x]#[name[i]] # Replace 'comment' with the actual column name
            actual_label = labeldf[label[i]].iloc[x]#[label[j]] # Replace 'label' with the actual column name
            print(f"Predictor: {predictor}")
            print(f"Actual label: {actual_label}")
            print("\n")
        print("No more misclassified examples.")
        probabilities = nb_model.predict_proba(np.array(X_test))
        predictions = probabilities[:, 1]
        fpr, tpr, threshold = roc_curve(y_test, predictions)
        roc_auc = auc(fpr, tpr)

        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
    



#Try combining


end = time.time()
print("total",end-begin)
