#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 23:36:47 2018

@author: jeff
"""

#investor similarity
'''
1b to 1d:
1b) Complete the investor profile by calculating the Profit & Loss
1c) Clustering for investors
1d) Output clustering results

P&L, industry


'''
'''************************
1b) Prepare investor Profile
'''
#load relevant libraries
import os
import pandas as pd
import numpy as np
from time import time
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

np.random.seed(42)

list_fld = ['investorname_x','calendardate_x','scale','action']
measure_fld =['value_x','value_y','realized_return','unrealized_return','new_money']
current_file_dir = os.path.dirname(__file__)

#Summarize quarterly performance of investors per quarter
input_path = os.path.join(current_file_dir,'data','investor_data')
file_list = os.listdir(input_path)
file_list = sorted(file_list)
investor_pd = pd.DataFrame()
for file in file_list:
    print(file)
    if not file.endswith('.csv'):
        continue
    file_path = os.path.join(input_path,file)
    tmp_pd = pd.read_csv(file_path)
    tmp_pd['unrealized_return']=0
    tmp_pd['realized_return']=0
    tmp_pd['new_money']=0
    for index, row in tmp_pd.iterrows():
        #units_chg = row['units_x']-row['units_y']
        
        if row['units_chg'] > 0:
            realized_return = 0
            unrealized_return = row['units_y']*row['price_chg']
            new_money = row['units_chg'] * row['price_x']
        else: #sell off or hold
            realized_return = (-row['units_chg'])*(row['price_chg'])
            unrealized_return = row['units_y']*(row['price_chg'])
            new_money = 0
        tmp_pd.loc[index,'unrealized_return']=unrealized_return
        tmp_pd.loc[index,'realized_return']=realized_return
        tmp_pd.loc[index,'new_money']=realized_return
        
    #calculate return (realized, unrealized and new money)
    if len(tmp_pd)>0:
        tmp_pd_group = tmp_pd.groupby(list_fld)[measure_fld].sum()
        tmp_pd_group['return'] = (tmp_pd_group['realized_return']+tmp_pd_group['unrealized_return'])/tmp_pd_group['value_y']
        tmp_pd_group['unrealized_return'] = (tmp_pd_group['unrealized_return'])/tmp_pd_group['value_y']
        tmp_pd_group['realized_return'] = (tmp_pd_group['realized_return'])/tmp_pd_group['value_y']
        tmp_pd_group['new_money'] = (tmp_pd_group['new_money'])/tmp_pd_group['value_y']
        tmp_pd_pivot = tmp_pd_group.pivot_table(values=['value_x','realized_return','return','unrealized_return','new_money']
            ,index =['investorname_x'],columns=['scale','action'],aggfunc={'return':np.mean,'return':np.std,
            'realized_return':np.mean,'realized_return':np.std,
            'unrealized_return':np.mean,'unrealized_return':np.std,
            'value_x':np.sum})
        #tmp_pd_pivot = tmp_pd_group.pivot_table(values=['return'],index =['investorname_x'],columns=['scale','action'],aggfunc={'return':np.mean})
        investor_pd = investor_pd.append(tmp_pd_pivot)

investor_pd.to_csv('investor_summary.csv')

'''************************
1c) Cluster investors
'''
#cleansed and transform data for clustering
investor_pd = investor_pd.replace([np.inf, ], 999999999) 
investor_pd = investor_pd.fillna(0)
investor_pd = investor_pd.dropna()

sc_X = StandardScaler()
X = sc_X.fit_transform(investor_pd)

#define the k means function
def bench_k_means(estimator, name, data):
    t0 = time()
    cluster_labels = estimator.fit_predict(data)
    score = metrics.silhouette_score(data, cluster_labels, metric='euclidean')
    t1 = time()
    print('time spent :' +str(t1-t0))
    return score,cluster_labels

#try out different K means parameters and find out the best parameters
best_score = 1
best_cluster = 0
best_labels = pd.DataFrame()
track ={}
best_KMeans_model = KMeans()
for num_cluster in range(5, 500):
    KMeans_model = KMeans(init='k-means++', n_clusters=num_cluster, n_init=10)
    this_score,this_labels = bench_k_means(KMeans_model,
              name="k-means++", data=X)
    track[num_cluster] = this_score
    if this_score < best_score:
        best_score = this_score
        best_cluster = num_cluster
        best_KMeans_model = KMeans_model
        best_labels = this_labels
        print(num_cluster)

'''************************
1d) Output the results
'''
#TO RUN
best_labels_pd = pd.DataFrame(best_labels)
best_labels_pd.columns = ['cluster']
X_pd = pd.DataFrame(X)
best_labels_data = pd.concat([X_pd,best_labels_pd],axis=1)

#Output clusters
f_cluster=open('investor_cluster_'+str(best_cluster)+'.pkl',"wb+")
pickle.dump(best_KMeans_model, f_cluster)
f_cluster.close()

f_SC=open('investor_SC_'+str(best_cluster)+'.pkl',"wb+")
pickle.dump(sc_X, f_SC)
f_SC.close()

f_labels=open('investor_labels_'+str(best_cluster)+'.pkl',"wb+")
pickle.dump(best_labels_data, f_labels)
f_labels.close()
