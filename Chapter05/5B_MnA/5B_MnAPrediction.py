#!/usr/bin/env python3
# -*- coding: utf-8 -*-
QUANDLKEY = '<Enter your Quandl APT key here>'
"""
Created on Fri Oct  5 23:24:35 2018

@author: jeff
"""
'''*************************************
#1. Import libraries and define key variables
'''
import pandas as pd
import numpy as np
import quandl
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report,roc_curve, auc,confusion_matrix,f1_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import graphviz

#KPI keys
quandl.ApiConfig.api_key = QUANDLKEY


'''*************************************
#2. Definition of functions
'''
#2a.Download tickers
def download_tkr(tkr):

    record_db_events_gp = pd.DataFrame()
    record_db_financials=quandl.get_table('SHARADAR/SF1', calendardate={'gte': '2008-12-31'}, ticker=tkr, dimension='MRY')    
    record_db_financials['year'] = record_db_financials['reportperiod'].dt.year
    record_db_financials['year_1'] = record_db_financials['year']+1
    
    record_db_events=quandl.get_table('SHARADAR/EVENTS', ticker=tkr)
    tmp_series = record_db_events['eventcodes'].str.contains('21')
    record_db_events= record_db_events[tmp_series]
    record_db_events['year'] = record_db_events.date.dt.year
    record_db_events= record_db_events.drop(['date'],axis=1)
    record_db_events_gp = record_db_events.groupby(['ticker','year'],as_index=False).count()
    
    combined_pd = pd.merge(record_db_financials,record_db_events_gp,how ='left',left_on='year_1',right_on='year')
    #convert all events to 1 and NaN
    combined_pd.loc[combined_pd['eventcodes']>1,'eventcodes'] = 1
    X = record_db_financials.iloc[:,6:-5]
    Y = combined_pd.iloc[:,-1]
    
    return combined_pd, X, Y
#tkr = 'AMZN'
#df_tmp = download_tkr(tkr)

#2b.Train tree
def train_tree(X,Y,ind):
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
    plt.title('Receiver operating characteristic - Decision Tree '+ind)
    plt.legend(loc="lower right")
    plt.savefig(ind+'_DT.png')

    ##metric 2: Confusion matrix
    Y_pred_tree = tree_clf.predict(X_test)
    confusion_matrix_tree = confusion_matrix(Y_test, Y_pred_tree)
    print(confusion_matrix_tree)
    print(classification_report(Y_test, Y_pred_tree))

    #common standard to compare across models
    f1_clf = f1_score(Y_test, Y_pred_tree, average='weighted')

    ##save model
    f_tree = open(ind+'_tree_clf.pkl',"wb+")
    pickle.dump(tree_clf, f_tree)
    f_tree.close()
    
    f_tree_sc = open(ind+'_tree_scaler.pkl',"wb+")
    pickle.dump(scaler, f_tree_sc)
    f_tree_sc.close()
    
    return tree_clf,f1_clf

##2C Neural Network
#2Ci. Grid search that simulate the performance of different neural network design
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
            temp_f1 = f1_score(Y_test, Y_pred, average='weighted')
            if temp_f1 > best_f1:
                best_f1 = temp_f1
                best_hidden_layers_list = hidden_layers_list
                best_hidden_layers_tuple = hidden_layers_tuple
    print(best_hidden_layers_list)
    return best_hidden_layers_list,best_hidden_layers_tuple

#2Cii. Train Neural Network
def train_NN(X,Y,ind):
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
    plt.title('Receiver operating characteristic - Neural Network '+ind)
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(ind+'_NN.png')

    ##metric 2: Confusion matrix
    Y_pred_tree = nn_clf.predict(X_test)
    confusion_matrix_tree = confusion_matrix(Y_test, Y_pred_tree)
    print(confusion_matrix_tree)
    print(classification_report(Y_test, Y_pred_tree))

    #common standard to compare across models
    #f1_clf = f1_score(Y_test, Y_score_nn, average='binary')
    f1_clf = f1_score(Y_test, Y_score_nn, average='weighted')

    ##save model
    f_nn = open(ind+'_nn_clf_.pkl',"wb+")
    pickle.dump(nn_clf, f_nn)
    f_nn.close()

    f_nn_sc = open(ind+'_nn_scaler.pkl',"wb+")
    pickle.dump(scaler, f_nn_sc)
    f_nn_sc.close()
    
    return nn_clf, f1_clf

'''*************************************
3. Execute the program
#3a. filter the industry in scope
'''
groupby_fld = 'sicsector'
min_size = 30
df_tkr = pd.read_csv('industry_tickers_list.csv')
dict_ind_tkr = {}
f1_list = []

df_tkr_ind = pd.DataFrame()
df_tkr_ind['cnt'] = df_tkr.groupby(groupby_fld)['ticker'].count()
df_tkr_ind_select = df_tkr_ind[df_tkr_ind['cnt']>=min_size]
list_scope = list(df_tkr_ind_select.index)

#collect ticker in each industry
for index, row in df_tkr.iterrows():
    ind = row[groupby_fld]
    tkr = row['ticker']
    if ind in list_scope:
        if ind in dict_ind_tkr:
            dict_ind_tkr[ind].append(tkr)
        else:
            dict_ind_tkr[ind] = [tkr]

#loop through the dictionary - one industry at a time
for ind, list_tkr in dict_ind_tkr.items():
    df_X = pd.DataFrame({})
    df_Y = pd.DataFrame({})
    print(ind)
    #Go through the ticker list to Download data from source
    #loop through tickers from that industry
    for tkr in list_tkr:
        print(tkr)
        try:
            df_tmp,X_tmp,Y_tmp = download_tkr(tkr)
        except Exception:
            continue
        
        if len(df_X)==0:
            #df_all = df_tmp
            df_X = X_tmp
            df_Y = Y_tmp
        else:
            #df_all = pd.concat([df_all,df_tmp])
            df_X = pd.concat([df_X,X_tmp])
            df_Y = pd.concat([df_Y,Y_tmp])

    ''' 
    *************************************
    3b. prepare features for clustering for the industry
    '''
    #convert to float and calc the difference across rows
    df_X = df_X.astype(float)
    df_Y = df_Y.astype(float)
            
    #remove zero records
    df_X = df_X.replace([np.inf ], 999999999) 
    df_X = df_X.fillna(0)
    df_Y = df_Y.fillna(0)
    
    #neural network
    nn_clf,f1_score_temp = train_NN(df_X,df_Y,ind)
    f1_list.append(f1_score_temp)
    nn_clf.get_params()
   
    #decision tree
    try:
        tree_clf,f1_score_temp = train_tree(df_X,df_Y,ind)
    except Exception:
        continue
    
    f1_list.append(f1_score_temp)
    tree_clf.get_params()

    '''
    #3c. Visualize the result
    '''
    fields_list = df_tmp.columns
    
    print('********************')
    print('f1 of the models')
    print(f1_list)
    print('********************')
    
    #for visualization of decision tree
    x_feature_name = fields_list[6:-8]
    y_target_name = fields_list[-1]
    d_tree_out_file = 'decision_tree_'+ind
    dot_data = tree.export_graphviz(tree_clf, out_file=None, 
                             feature_names=x_feature_name,  
                             class_names=y_target_name,  
                             filled=True, rounded=True,  
                             special_characters=True) 
    graph = graphviz.Source(dot_data) 
    graph.render(d_tree_out_file)
