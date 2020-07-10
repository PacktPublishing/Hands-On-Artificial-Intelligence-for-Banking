#!/usr/bin/env python3
# -*- coding: utf-8 -*-
QUANDLKEY = '<Enter your Quandl APT key here>'
"""
Created on Sun Sep 30 01:00:02 2018

@author: jeff
"""
'''*************************************
i. load industry, tickers and functions
'''
#import libraries
import quandl
import pandas as pd
import numpy as np
import os
from time import time
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

#KPI keys
quandl.ApiConfig.api_key = QUANDLKEY

current_file_dir = os.path.dirname(__file__)
#define important functions
#download fundamental data of the ticker
def download_tkr(tkr):
    record_db_t_2017Q4=quandl.get_table('SHARADAR/SF1', calendardate='2017-12-31', ticker=tkr,dimension='MRY')    
    list_all = [record_db_t_2017Q4]
    return list_all

#kmean clustering function
def bench_k_means(estimator, name, data):
    t0 = time()
    cluster_labels = estimator.fit_predict(data)
    score = metrics.silhouette_score(data, cluster_labels, metric='euclidean')
    t1 = time()
    print('time spent :' +str(t1-t0))
    return score,cluster_labels


'''*************************************
#2a. load data
'''
#parameters
groupby_fld = 'sicindustry'
min_size = 30
df_tkr = pd.read_csv('industry_tickers_list.csv')
f_para = open('output_para.csv','w+')
dict_ind_tkr = {}

'''*************************************
#i. filter the industry in scope
'''
df_tkr_ind = pd.DataFrame()
df_tkr_ind['cnt'] = df_tkr.groupby(groupby_fld)['ticker'].count()
df_tkr_ind_select = df_tkr_ind[df_tkr_ind['cnt']>=min_size]
list_scope = list(df_tkr_ind_select.index)

#collect tkr in each industry
for index, row in df_tkr.iterrows():
    ind = row[groupby_fld]
    tkr = row['ticker']
    if ind in list_scope:
        if ind in dict_ind_tkr:
            dict_ind_tkr[ind].append(tkr)
        else:
            dict_ind_tkr[ind] = [tkr]

'''*************************************
#ii. create a dataframe for each industry to do clustering
'''
list_fld = ['ticker','ps1','pe1','pb'
            ,'marketcap'
            ,'divyield','bvps'
            ,'ebitda','grossmargin','netmargin','roa','roe'
            ,'de']
target_score = 0.05
#loop through the industry
for ind, list_tkr in dict_ind_tkr.items():
    df_all = pd.DataFrame({})
    print(ind)
    #Go through the ticker list to Download data from source
    #loop through tickers from that industry
    for tkr in list_tkr:
        print(tkr)
        try:
            list_all = download_tkr(tkr)
        except Exception:
            next
        df_tmp = pd.concat(list_all)[list_fld]
        if len(df_all)==0:
            df_all = df_tmp
        else:
            df_all = pd.concat([df_all,df_tmp])
    
    '''*************************************
    2b. prepare features for clustering for the industry
    '''
    #convert to float and calc the difference across rows
    df_all.index = df_all['ticker']
    df_all = df_all.drop(['ticker'],axis=1)
    df_fs = df_all.astype(float)
            
    #remove zero records
    df_fs = df_fs.replace([np.inf ], 999999999) 
    df_fs = df_fs.fillna(0)
    df_fs_filter=df_fs.dropna()

    if len(df_fs_filter) == 0:
        continue
    '''*************************************
    2C. Perform K means clustering for the industry
    '''
    #clustering        
    sc_X = StandardScaler()
    X = sc_X.fit_transform(df_fs_filter)

    best_score = 1
    best_cluster = 0
    best_labels = pd.DataFrame()
    track ={}
    best_KMeans_model = KMeans()
    df_fs_filter_tickers = pd.DataFrame()
    df_fs_filter_tickers['ticker'] = df_fs_filter.index
    df_fs_filter_tickers=df_fs_filter_tickers.reset_index()
    max_clsuter = int(len(df_fs_filter)/2)
    for num_cluster in range(5, max_clsuter):
        KMeans_model = KMeans(init='k-means++', n_clusters=num_cluster, n_init=10)
        this_score,this_labels = bench_k_means(KMeans_model,
                  name="k-means++", data=X)
        track[num_cluster] = this_score
        if this_score < best_score:
            best_score = this_score
            best_cluster = num_cluster
            best_KMeans_model = KMeans_model
            labels_df = pd.DataFrame(this_labels)
            labels_df.columns = ['cluster']
            best_labels = pd.concat([df_fs_filter_tickers,labels_df],axis=1)
            print(num_cluster)
        if best_score <= target_score:
            break

    '''*************************************
    2D. Output the clustering model and scaler for the industry
    '''    
    #Output clusters
    file_path = os.path.join(current_file_dir,'data','stock_data', ind+'_cluster_'+str(best_cluster)+'.pkl')
    f_cluster=open(file_path,"wb+")
    pickle.dump(best_KMeans_model, f_cluster)
    f_cluster.close()
    
    file_path = os.path.join(current_file_dir,'data','stock_data',ind+'_SC_'+str(best_cluster)+'.pkl')
    f_SC=open(file_path,"wb+")
    pickle.dump(sc_X, f_SC)
    f_SC.close()
    
    file_path = os.path.join(current_file_dir,'data','stock_data',ind+'_labels_'+str(best_cluster)+'.pkl')
    f_labels=open(file_path,"wb+")
    pickle.dump(best_labels, f_labels)
    f_labels.close()
