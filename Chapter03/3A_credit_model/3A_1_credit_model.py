#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 18:10:37 2018

@author: jeff

special thanks to haloboy777 on converting the arff dataset to csv
#########################################
# Project   : ARFF to CSV converter     #
# Created   : 10/01/17 11:08:06         #
# Author    : haloboy777                #
# Licence   : MIT                       #
#########################################

"""

'''*************************************
1. Import libraries and define key variables
'''
import os
import re
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.metrics import classification_report,roc_curve, auc,confusion_matrix,f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn import linear_model,tree
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import pickle
import graphviz

'''*************************************
1. Define program-wide variables and values
'''
#set up the working directory where datafiles are all located
data_path = os.getcwd()
os.chdir(data_path)

#read in files
file_attrib_in = os.path.join(data_path,'attrib.txt')
file_path_in = os.path.join(data_path,'5year.csv')
file_path_out = os.path.join(data_path,'output_dataset.txt')
file_corr_out = os.path.join(data_path,'corr.txt')
f_name = pd.read_csv(file_path_in,sep=',')
f_attrib = open(file_attrib_in,"r")
attrib_str = f_attrib.read()

#assign column headers
label_name = 'default'
re_obj = re.compile(r'X[0-9]+\s')
fields_list = re_obj.split(attrib_str)
fields_list = fields_list[1:]
fields_list.append(label_name)
f_name.columns = fields_list

#create X and Y dataset with the right header
X = f_name.iloc[:,:-1]
Y= f_name.iloc[:,-1]
#make sure data types are correct and missing values are handled
Y=Y.astype(int)
cols = X.columns[X.dtypes.eq(object)]
for c in cols:
    X[c] = pd.to_numeric(X[c], errors='coerce')
X=X.fillna(0)

'''
2. Define all the functions
'''
'''
##2A. Logistic Regression model
'''
##2A.i
#input the dataframe and the list of columns wanted from it, it return the dataframe with columns selected
def select_columns(df, col_list):
    df_selected = df[df.columns.intersection(col_list)]
    return df_selected

##2A.ii
#input the support (true/false) of the column lists, and the column header, return the list only with true value
def generate_column_lists(col_support,col_list):
    i = 0
    select_cols = []
    len_list = len(col_list)
    while i< len_list:
        if col_support[i]:
            select_cols.append(col_list[i])
        i=i+1
    return select_cols

##2A.iii
##try any number of features, return the #of features that deliver the best accuracy (AUC)
def optimize_RFE(logreg, X, Y, target_features = 10):
    trial_cnt = 1
    max_roc_auc=0
    #best_feature = 0
    best_col_list = []
    result_list = {}
    col_list = list(X.columns.values)

    while trial_cnt<=target_features:
        rfe = RFE(logreg,trial_cnt,verbose=1)
        rfe = rfe.fit(X,Y)
        print(rfe.support_)
        print(rfe.ranking_)
        col_support = rfe.support_
    
        #select the columns
        select_cols = generate_column_lists(col_support, col_list)

        #generate the dataframe with only the list of columns
        X_selected = select_columns(X,select_cols)
        print(list(X_selected.columns))
            
        #build model
        print('split data')
        X_train, X_test, Y_train, Y_test = train_test_split(X_selected, Y, test_size=0.33, random_state=42)
        print('build model')
        logreg.fit(X_train,Y_train)
        Y_score = logreg.decision_function(X_test)
    
        ##metric 1: roc
        fpr, tpr, thresholds = roc_curve(Y_test,Y_score, pos_label=1)
        roc_auc = auc(fpr,tpr)
        
        result_list[trial_cnt] = roc_auc
        result_list['F_'+str(trial_cnt)] = select_cols
        
        #memorize this setting if this ROC is the highest
        if roc_auc > max_roc_auc:
            max_roc_auc = roc_auc
            #best_feature = trial_cnt
            best_col_list = select_cols
            print('roc_updated at '+ str(trial_cnt))            
        
        trial_cnt=trial_cnt+1
        
    return max_roc_auc, best_col_list, result_list

##2A.iv
#feed in data to the logistic regression model
def train_logreg(X,Y):
    print('Logistic Regression')
    logreg = linear_model.LogisticRegression(C=1e5)
    #find out the features that deliver the highest accuracy
    #roc_auc, best_col_list, result_list = optimize_RFE(logreg, X,Y,len(X.columns)-1)
    
    roc_auc, best_col_list, result_list = optimize_RFE(logreg, X,Y,20)
    
    #split the dataset into training set and testing set
    X_selected = select_columns(X, best_col_list)
    X_train, X_test, Y_train, Y_test = train_test_split(X_selected, Y, test_size=0.33, random_state=42)
    
    #fit the training data to the model
    #preprocessing the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test) 
    logreg.fit(X_train,Y_train)

    ##metric 1: roc
    Y_score_logreg = logreg.decision_function(X_test)
    fpr, tpr, thresholds = roc_curve(Y_test,Y_score_logreg, pos_label=1)
    roc_auc = auc(fpr,tpr)
    lw=2
    plt.figure()
    plt.plot(fpr,tpr,color='darkorange',lw=lw,label='ROC curve (area = %0.2f)' %roc_auc)
    plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic - Logistics Regression Model')
    plt.legend(loc="lower right")
    plt.show()

    ##metric 2: Confusion matrix
    Y_pred_logreg = logreg.predict(X_test)
    confusion_matrix_logreg = confusion_matrix(Y_test, Y_pred_logreg)
    print(confusion_matrix_logreg)
    print(classification_report(Y_test, Y_pred_logreg))

    #common standard to compare across models
    f1_clf = f1_score(Y_test, Y_pred_logreg, average='binary')

    ##Quality Check: tets for depedency
    corr_m = X_selected.corr()
    sns.heatmap(corr_m)
    corr_m.to_csv(file_corr_out)
    plt.show()

    ##save model
    f_logreg=open('log_reg.pkl',"wb+")
    pickle.dump(logreg, f_logreg)
    f_logreg.close()

    f_logreg_sc = open('logreg_scaler.pkl',"wb+")
    pickle.dump(scaler, f_logreg_sc)
    f_logreg_sc.close()

    print('These columns are in the final model')
    print(best_col_list)
    thefile = open('logreg_cols.txt', 'w+')
    for item in best_col_list:
        thefile.write("%s\n" % item)    
    '''
    [[1790   21]
    [ 118   22]]
             precision    recall  f1-score   support

          0       0.94      0.99      0.96      1811
          1       0.51      0.16      0.24       140

    avg / total       0.91      0.93      0.91      1951
    '''
    return logreg, f1_clf

'''
##2B. Decision Tree
'''
##2B.i
#feed in data to the decision tree
def train_tree(X,Y):
    print('Decision Tree')
    #split the dataset into training set and testing set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33, random_state=0)
    
    min_leaf_size = int(len(X_train) * 0.01)
    tree_clf = tree.DecisionTreeClassifier(min_samples_leaf=min_leaf_size)
 
    #preprocessing the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)    
    
    #fit the training data to the model
    tree_clf.fit(X_train,Y_train)

    ##metric 1: roc
    Y_score_tree = tree_clf.predict(X_test)
    fpr, tpr, thresholds = roc_curve(Y_test,Y_score_tree, pos_label=1)
    roc_auc = auc(fpr,tpr)
    lw=2
    plt.figure()
    plt.plot(fpr,tpr,color='darkorange',lw=lw,label='ROC curve (area = %0.2f)' %roc_auc)
    plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic - Decision Tree')
    plt.legend(loc="lower right")
    plt.show()

    ##metric 2: Confusion matrix
    Y_pred_tree = tree_clf.predict(X_test)
    confusion_matrix_tree = confusion_matrix(Y_test, Y_pred_tree)
    print(confusion_matrix_tree)
    print(classification_report(Y_test, Y_pred_tree))

    #common standard to compare across models
    f1_clf = f1_score(Y_test, Y_pred_tree, average='binary')

    ##save model
    f_tree = open('tree_clf.pkl',"wb+")
    pickle.dump(tree_clf, f_tree)
    f_tree.close()
    
    f_tree_sc = open('tree_scaler.pkl',"wb+")
    pickle.dump(scaler, f_tree_sc)
    f_tree_sc.close()
    '''
    [[1801   27]
    [  62   61]]
             precision    recall  f1-score   support

          0       0.97      0.99      0.98      1828
          1       0.69      0.50      0.58       123

    avg / total       0.95      0.95      0.95      1951
    '''
 
    
    return tree_clf,f1_clf

##2C Neural Network
##2Ci. Grid search that simulate the performance of different neural network design
def grid_search(X_train,X_test, Y_train,Y_test,num_training_sample):
    
    best_f1 = 0
    best_hidden_layers_list = []
    best_hidden_layers_tuple = ()
    #various depth
    for depth in range(1,5):
        print('Depth = '+str(depth))
        for layer_size in range(1,8):
            neuron_cnt = 0
            hidden_layers_list = []        
            i = 0
            while i<depth:
                hidden_layers_list.append(layer_size)
                neuron_cnt += layer_size
                i+=1
            #pruning - to avoid over-training
            if num_training_sample<neuron_cnt:
                break
            
            hidden_layers_tuple = tuple(hidden_layers_list)
            nn_clf = MLPClassifier(alpha=1e-5,
                     hidden_layer_sizes=hidden_layers_tuple, random_state=1)
            
            nn_clf.fit(X_train,Y_train)
            Y_pred = nn_clf.predict(X_test)
            temp_f1 = f1_score(Y_test, Y_pred, average='binary')
            if temp_f1 > best_f1:
                best_f1 = temp_f1
                best_hidden_layers_list = hidden_layers_list
                best_hidden_layers_tuple = hidden_layers_tuple
    print(best_hidden_layers_list)
    return best_hidden_layers_list,best_hidden_layers_tuple
                
    #various size
# referencing: https://www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/
##2Cii. train network network
def train_NN(X,Y):
    print('Neural Network')
    #split the dataset into training set and testing set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33, random_state=0)
    
    #preprocessing the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    num_training_sample = len(X_train)
    best_hidden_layers_list,best_hidden_layers_tuple = grid_search(X_train, X_test, Y_train, Y_test,num_training_sample)
    nn_clf = MLPClassifier(alpha=1e-5,
                     hidden_layer_sizes=best_hidden_layers_tuple, random_state=1)
    
    #fit the training data to the model
    nn_clf.fit(X_train,Y_train)

    ##metric 1: roc
    Y_score_nn = nn_clf.predict(X_test)
    fpr, tpr, thresholds = roc_curve(Y_test,Y_score_nn, pos_label=1)
    roc_auc = auc(fpr,tpr)
    lw=2
    plt.figure()
    plt.plot(fpr,tpr,color='darkorange',lw=lw,label='ROC curve (area = %0.2f)' %roc_auc)
    plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic - Neural Network')
    plt.legend(loc="lower right")
    plt.show()

    ##metric 2: Confusion matrix
    Y_pred_tree = nn_clf.predict(X_test)
    confusion_matrix_tree = confusion_matrix(Y_test, Y_pred_tree)
    print(confusion_matrix_tree)
    print(classification_report(Y_test, Y_pred_tree))

    #common standard to compare across models
    f1_clf = f1_score(Y_test, Y_score_nn, average='binary')

    ##save model
    f_nn = open('nn_clf.pkl',"wb+")
    pickle.dump(nn_clf, f_nn)
    f_nn.close()

    f_nn_sc = open('nn_scaler.pkl',"wb+")
    pickle.dump(scaler, f_nn_sc)
    f_nn_sc.close()

    '''
    [[1808   20]
     [  85   38]]
                 precision    recall  f1-score   support
    
              0       0.96      0.99      0.97      1828
              1       0.66      0.31      0.42       123
    
    avg / total       0.94      0.95      0.94      1951
    '''
    
    return nn_clf, f1_clf

'''
3. Run the functions above
'''
f1_list = []
f1_score_temp= 0
#logistic regression model
log_reg,f1_score_temp = train_logreg(X,Y)
f1_list.append(f1_score_temp)
log_reg.get_params()

#decision tree
tree_clf,f1_score_temp = train_tree(X,Y)
f1_list.append(f1_score_temp)
tree_clf.get_params()
#neural network
nn_clf,f1_score_temp = train_NN(X,Y)
f1_list.append(f1_score_temp)
nn_clf.get_params()

'''
#4 Visualize the result
'''
print('********************')
print('f1 of the models')
print(f1_list)
print('********************')

#for visualization of decision tree
x_feature_name = fields_list[:-1]
y_target_name = fields_list[-1]
d_tree_out_file = 'decision_tree'
dot_data = tree.export_graphviz(tree_clf, out_file=None, 
                         feature_names=x_feature_name,  
                         class_names=y_target_name,  
                         filled=True, rounded=True,  
                         special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.render(d_tree_out_file) 
