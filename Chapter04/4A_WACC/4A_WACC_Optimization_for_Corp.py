#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 22:46:06 2018

@author: jeff
"""
QUANDLKEY = '<Enter your Quandl APT key here>'
'''
subscription to 'SHARADAR/SEP' and 'SHARADAR/SF1' of quandl database
take the 'log_reg.pkl' and 'logreg_scaler.pkl' from 3A_1_credit_model to this file's directory
'''
import quandl
import pickle
import numpy as np
import math
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt 
#pip3 install -U Pillow in case you encounter error with PIL
import seaborn as sns

'''*************************************
1. Download & Load Data
'''
print('Retrieve data')
tkr = 'DUK'
quandl.ApiConfig.api_key = QUANDLKEY

'''*************************************
1A. Retreive data for 2A.
'''
econ = quandl.get("FRED/TEDRATE", authtoken=QUANDLKEY, start_date='2018-05-31', end_date='2018-07-31')
NYSE_index = quandl.get('WFE/INDEXES_NYSECOMPOSITE', start_date='2013-05-31', end_date='2018-07-31')

'''*************************************
1B. Retrieve Data for the target ticker
'''
record_db = quandl.get_table('SHARADAR/SF1', calendardate='2017-12-31', ticker=tkr,dimension='MRY')
record_db_t_1 = quandl.get_table('SHARADAR/SF1', calendardate='2016-12-31', ticker=tkr,dimension='MRY')

'''*************************************
1C. Download & Load Data for 2C.
'''
tkr = 'DUK'
quandl.ApiConfig.api_key = QUANDLKEY
record_db_t_2017Q1=quandl.get_table('SHARADAR/SF1', calendardate='2017-3-31', ticker=tkr,dimension='MRQ')
record_db_t_2017Q2=quandl.get_table('SHARADAR/SF1', calendardate='2017-6-30', ticker=tkr,dimension='MRQ')
record_db_t_2017Q3=quandl.get_table('SHARADAR/SF1', calendardate='2017-9-30', ticker=tkr,dimension='MRQ')
record_db_t_2017Q4=quandl.get_table('SHARADAR/SF1', calendardate='2017-12-31', ticker=tkr,dimension='MRQ')
record_db_t_2016Q1=quandl.get_table('SHARADAR/SF1', calendardate='2016-3-31', ticker=tkr,dimension='MRQ')
record_db_t_2016Q2=quandl.get_table('SHARADAR/SF1', calendardate='2016-6-30', ticker=tkr,dimension='MRQ')
record_db_t_2016Q3=quandl.get_table('SHARADAR/SF1', calendardate='2016-9-30', ticker=tkr,dimension='MRQ')
record_db_t_2016Q4=quandl.get_table('SHARADAR/SF1', calendardate='2016-12-31', ticker=tkr,dimension='MRQ')
record_db_t_2015Q1=quandl.get_table('SHARADAR/SF1', calendardate='2015-3-31', ticker=tkr,dimension='MRQ')
record_db_t_2015Q2=quandl.get_table('SHARADAR/SF1', calendardate='2015-6-30', ticker=tkr,dimension='MRQ')
record_db_t_2015Q3=quandl.get_table('SHARADAR/SF1', calendardate='2015-9-30', ticker=tkr,dimension='MRQ')
record_db_t_2015Q4=quandl.get_table('SHARADAR/SF1', calendardate='2015-12-31', ticker=tkr,dimension='MRQ')
record_db_t_2014Q1=quandl.get_table('SHARADAR/SF1', calendardate='2014-3-31', ticker=tkr,dimension='MRQ')
record_db_t_2014Q2=quandl.get_table('SHARADAR/SF1', calendardate='2014-6-30', ticker=tkr,dimension='MRQ')
record_db_t_2014Q3=quandl.get_table('SHARADAR/SF1', calendardate='2014-9-30', ticker=tkr,dimension='MRQ')
record_db_t_2014Q4=quandl.get_table('SHARADAR/SF1', calendardate='2014-12-31', ticker=tkr,dimension='MRQ')
record_db_t_2013Q1=quandl.get_table('SHARADAR/SF1', calendardate='2013-3-31', ticker=tkr,dimension='MRQ')
record_db_t_2013Q2=quandl.get_table('SHARADAR/SF1', calendardate='2013-6-30', ticker=tkr,dimension='MRQ')
record_db_t_2013Q3=quandl.get_table('SHARADAR/SF1', calendardate='2013-9-30', ticker=tkr,dimension='MRQ')
record_db_t_2013Q4=quandl.get_table('SHARADAR/SF1', calendardate='2013-12-31', ticker=tkr,dimension='MRQ')
record_db_t_2012Q1=quandl.get_table('SHARADAR/SF1', calendardate='2012-3-31', ticker=tkr,dimension='MRQ')
record_db_t_2012Q2=quandl.get_table('SHARADAR/SF1', calendardate='2012-6-30', ticker=tkr,dimension='MRQ')
record_db_t_2012Q3=quandl.get_table('SHARADAR/SF1', calendardate='2012-9-30', ticker=tkr,dimension='MRQ')
record_db_t_2012Q4=quandl.get_table('SHARADAR/SF1', calendardate='2012-12-31', ticker=tkr,dimension='MRQ')
record_db_t_2011Q1=quandl.get_table('SHARADAR/SF1', calendardate='2011-3-31', ticker=tkr,dimension='MRQ')
record_db_t_2011Q2=quandl.get_table('SHARADAR/SF1', calendardate='2011-6-30', ticker=tkr,dimension='MRQ')
record_db_t_2011Q3=quandl.get_table('SHARADAR/SF1', calendardate='2011-9-30', ticker=tkr,dimension='MRQ')
record_db_t_2011Q4=quandl.get_table('SHARADAR/SF1', calendardate='2011-12-31', ticker=tkr,dimension='MRQ')
record_db_t_2010Q1=quandl.get_table('SHARADAR/SF1', calendardate='2010-3-31', ticker=tkr,dimension='MRQ')
record_db_t_2010Q2=quandl.get_table('SHARADAR/SF1', calendardate='2010-6-30', ticker=tkr,dimension='MRQ')
record_db_t_2010Q3=quandl.get_table('SHARADAR/SF1', calendardate='2010-9-30', ticker=tkr,dimension='MRQ')
record_db_t_2010Q4=quandl.get_table('SHARADAR/SF1', calendardate='2010-12-31', ticker=tkr,dimension='MRQ')
record_db_t_2009Q1=quandl.get_table('SHARADAR/SF1', calendardate='2009-3-31', ticker=tkr,dimension='MRQ')
record_db_t_2009Q2=quandl.get_table('SHARADAR/SF1', calendardate='2009-6-30', ticker=tkr,dimension='MRQ')
record_db_t_2009Q3=quandl.get_table('SHARADAR/SF1', calendardate='2009-9-30', ticker=tkr,dimension='MRQ')
record_db_t_2009Q4=quandl.get_table('SHARADAR/SF1', calendardate='2009-12-31', ticker=tkr,dimension='MRQ')
record_db_t_2008Q1=quandl.get_table('SHARADAR/SF1', calendardate='2008-3-31', ticker=tkr,dimension='MRQ')
record_db_t_2008Q2=quandl.get_table('SHARADAR/SF1', calendardate='2008-6-30', ticker=tkr,dimension='MRQ')
record_db_t_2008Q3=quandl.get_table('SHARADAR/SF1', calendardate='2008-9-30', ticker=tkr,dimension='MRQ')
record_db_t_2008Q4=quandl.get_table('SHARADAR/SF1', calendardate='2008-12-31', ticker=tkr,dimension='MRQ')
record_db_t_2007Q1=quandl.get_table('SHARADAR/SF1', calendardate='2007-3-31', ticker=tkr,dimension='MRQ')
record_db_t_2007Q2=quandl.get_table('SHARADAR/SF1', calendardate='2007-6-30', ticker=tkr,dimension='MRQ')
record_db_t_2007Q3=quandl.get_table('SHARADAR/SF1', calendardate='2007-9-30', ticker=tkr,dimension='MRQ')
record_db_t_2007Q4=quandl.get_table('SHARADAR/SF1', calendardate='2007-12-31', ticker=tkr,dimension='MRQ')
record_db_t_2006Q1=quandl.get_table('SHARADAR/SF1', calendardate='2006-3-31', ticker=tkr,dimension='MRQ')
record_db_t_2006Q2=quandl.get_table('SHARADAR/SF1', calendardate='2006-6-30', ticker=tkr,dimension='MRQ')
record_db_t_2006Q3=quandl.get_table('SHARADAR/SF1', calendardate='2006-9-30', ticker=tkr,dimension='MRQ')
record_db_t_2006Q4=quandl.get_table('SHARADAR/SF1', calendardate='2006-12-31', ticker=tkr,dimension='MRQ')
record_db_t_2005Q1=quandl.get_table('SHARADAR/SF1', calendardate='2005-3-31', ticker=tkr,dimension='MRQ')
record_db_t_2005Q2=quandl.get_table('SHARADAR/SF1', calendardate='2005-6-30', ticker=tkr,dimension='MRQ')
record_db_t_2005Q3=quandl.get_table('SHARADAR/SF1', calendardate='2005-9-30', ticker=tkr,dimension='MRQ')
record_db_t_2005Q4=quandl.get_table('SHARADAR/SF1', calendardate='2005-12-31', ticker=tkr,dimension='MRQ')
record_db_t_2004Q1=quandl.get_table('SHARADAR/SF1', calendardate='2004-3-31', ticker=tkr,dimension='MRQ')
record_db_t_2004Q2=quandl.get_table('SHARADAR/SF1', calendardate='2004-6-30', ticker=tkr,dimension='MRQ')
record_db_t_2004Q3=quandl.get_table('SHARADAR/SF1', calendardate='2004-9-30', ticker=tkr,dimension='MRQ')
record_db_t_2004Q4=quandl.get_table('SHARADAR/SF1', calendardate='2004-12-31', ticker=tkr,dimension='MRQ')
record_db_t_2003Q1=quandl.get_table('SHARADAR/SF1', calendardate='2003-3-31', ticker=tkr,dimension='MRQ')
record_db_t_2003Q2=quandl.get_table('SHARADAR/SF1', calendardate='2003-6-30', ticker=tkr,dimension='MRQ')
record_db_t_2003Q3=quandl.get_table('SHARADAR/SF1', calendardate='2003-9-30', ticker=tkr,dimension='MRQ')
record_db_t_2003Q4=quandl.get_table('SHARADAR/SF1', calendardate='2003-12-31', ticker=tkr,dimension='MRQ')
record_db_t_2002Q1=quandl.get_table('SHARADAR/SF1', calendardate='2002-3-31', ticker=tkr,dimension='MRQ')
record_db_t_2002Q2=quandl.get_table('SHARADAR/SF1', calendardate='2002-6-30', ticker=tkr,dimension='MRQ')
record_db_t_2002Q3=quandl.get_table('SHARADAR/SF1', calendardate='2002-9-30', ticker=tkr,dimension='MRQ')
record_db_t_2002Q4=quandl.get_table('SHARADAR/SF1', calendardate='2002-12-31', ticker=tkr,dimension='MRQ')
record_db_t_2001Q1=quandl.get_table('SHARADAR/SF1', calendardate='2001-3-31', ticker=tkr,dimension='MRQ')
record_db_t_2001Q2=quandl.get_table('SHARADAR/SF1', calendardate='2001-6-30', ticker=tkr,dimension='MRQ')
record_db_t_2001Q3=quandl.get_table('SHARADAR/SF1', calendardate='2001-9-30', ticker=tkr,dimension='MRQ')
record_db_t_2001Q4=quandl.get_table('SHARADAR/SF1', calendardate='2001-12-31', ticker=tkr,dimension='MRQ')
record_db_t_2000Q1=quandl.get_table('SHARADAR/SF1', calendardate='2000-3-31', ticker=tkr,dimension='MRQ')
record_db_t_2000Q2=quandl.get_table('SHARADAR/SF1', calendardate='2000-6-30', ticker=tkr,dimension='MRQ')
record_db_t_2000Q3=quandl.get_table('SHARADAR/SF1', calendardate='2000-9-30', ticker=tkr,dimension='MRQ')
record_db_t_2000Q4=quandl.get_table('SHARADAR/SF1', calendardate='2000-12-31', ticker=tkr,dimension='MRQ')
record_db_t_1999Q1=quandl.get_table('SHARADAR/SF1', calendardate='1999-3-31', ticker=tkr,dimension='MRQ')
record_db_t_1999Q2=quandl.get_table('SHARADAR/SF1', calendardate='1999-6-30', ticker=tkr,dimension='MRQ')
record_db_t_1999Q3=quandl.get_table('SHARADAR/SF1', calendardate='1999-9-30', ticker=tkr,dimension='MRQ')
record_db_t_1999Q4=quandl.get_table('SHARADAR/SF1', calendardate='1999-12-31', ticker=tkr,dimension='MRQ')
record_db_t_1998Q1=quandl.get_table('SHARADAR/SF1', calendardate='1998-3-31', ticker=tkr,dimension='MRQ')
record_db_t_1998Q2=quandl.get_table('SHARADAR/SF1', calendardate='1998-6-30', ticker=tkr,dimension='MRQ')
record_db_t_1998Q3=quandl.get_table('SHARADAR/SF1', calendardate='1998-9-30', ticker=tkr,dimension='MRQ')
record_db_t_1998Q4=quandl.get_table('SHARADAR/SF1', calendardate='1998-12-31', ticker=tkr,dimension='MRQ')


list_all = [record_db_t_2017Q4
,record_db_t_2017Q3
,record_db_t_2017Q2
,record_db_t_2017Q1
,record_db_t_2016Q4
,record_db_t_2016Q3
,record_db_t_2016Q2
,record_db_t_2016Q1
,record_db_t_2015Q4
,record_db_t_2015Q3
,record_db_t_2015Q2
,record_db_t_2015Q1
,record_db_t_2014Q4
,record_db_t_2014Q3
,record_db_t_2014Q2
,record_db_t_2014Q1
,record_db_t_2013Q4
,record_db_t_2013Q3
,record_db_t_2013Q2
,record_db_t_2013Q1
,record_db_t_2012Q4
,record_db_t_2012Q3
,record_db_t_2012Q2
,record_db_t_2012Q1
,record_db_t_2011Q4
,record_db_t_2011Q3
,record_db_t_2011Q2
,record_db_t_2011Q1
,record_db_t_2010Q4
,record_db_t_2010Q3
,record_db_t_2010Q2
,record_db_t_2010Q1
,record_db_t_2009Q4
,record_db_t_2009Q3
,record_db_t_2009Q2
,record_db_t_2009Q1
,record_db_t_2008Q4
,record_db_t_2008Q3
,record_db_t_2008Q2
,record_db_t_2008Q1
,record_db_t_2007Q4
,record_db_t_2007Q3
,record_db_t_2007Q2
,record_db_t_2007Q1
,record_db_t_2006Q4
,record_db_t_2006Q3
,record_db_t_2006Q2
,record_db_t_2006Q1
,record_db_t_2005Q4
,record_db_t_2005Q3
,record_db_t_2005Q2
,record_db_t_2005Q1
,record_db_t_2004Q4
,record_db_t_2004Q3
,record_db_t_2004Q2
,record_db_t_2004Q1
,record_db_t_2003Q4
,record_db_t_2003Q3
,record_db_t_2003Q2
,record_db_t_2003Q1
,record_db_t_2002Q4
,record_db_t_2002Q3
,record_db_t_2002Q2
,record_db_t_2002Q1
,record_db_t_2001Q4
,record_db_t_2001Q3
,record_db_t_2001Q2
,record_db_t_2001Q1
,record_db_t_2000Q4
,record_db_t_2000Q3
,record_db_t_2000Q2
,record_db_t_2000Q1
,record_db_t_1999Q4
,record_db_t_1999Q3
,record_db_t_1999Q2
,record_db_t_1999Q1
,record_db_t_1998Q4
,record_db_t_1998Q3
]
df_all = pd.concat(list_all)

#fix the dataframes
df_all.index = df_all['calendardate']
df_all = df_all.sort_index(ascending = False)
df_fs = df_all[['revenue','cor','sgna','opex','ppnenet','payables','receivables','inventory']]

#convert to float
df_fs_diff = df_fs
df_fs_diff = df_fs_diff.astype(float)
df_fs_diff = df_fs_diff.diff()

#create new fields
df_fs_diff.columns = ['revenue_diff','cor_diff','sgna_diff','opex_diff','ppnenet_diff','payables_diff','receivables_diff','inventory_diff']
df_fs_combine = pd.merge(df_fs,df_fs_diff,left_index = True, right_index = True)
df_fs_combine=df_fs_combine[1:]

#remove any record with na and 0 values to avoid division errors
df_fs_combine=df_fs_combine.dropna()

#we take a proxy here, should use last period's numbers as denominator not current period
df_fs_combine['revenue_chg']=df_fs_combine['revenue_diff']/df_fs_combine['revenue']
df_fs_combine['cor_chg']=df_fs_combine['cor_diff']/df_fs_combine['cor']
try:
    df_fs_combine['sgna_chg']=df_fs_combine['sgna_diff']/df_fs_combine['sgna']
except Exception:
    df_fs_combine['sgna_chg'] = 0
df_fs_combine['opex_chg']=df_fs_combine['opex_diff']/df_fs_combine['opex']
df_fs_combine['ppnenet_chg']=df_fs_combine['ppnenet_diff']/df_fs_combine['ppnenet']
df_fs_combine['payables_chg']=df_fs_combine['payables_diff']/df_fs_combine['payables']
df_fs_combine['receivables_chg']=df_fs_combine['receivables_diff']/df_fs_combine['receivables']
df_fs_combine['inventory_chg']=df_fs_combine['inventory_diff']/df_fs_combine['inventory']
'''*************************************
2A. Cost of Equity Formula
'''
print('CAPM')
#NUK and NYSE price
#calculate the CAPM:
list_price=[]
list_NYSE = []
i=0
prev_NYSE_val = 0
prev_price_val = 0
NYSE_val = 0
price_val = 0
for index, row in NYSE_index.iterrows():
    tmp_p = quandl.get_table('SHARADAR/SEP', date=index.date(), ticker=tkr)
    avg_p = (tmp_p['high']+tmp_p['low'])/2
    try:
        NYSE_val = avg_p.values[0]
    except Exception:
        next
    price_val =row['Value']
    if i>0:
        g_NYSE = NYSE_val/prev_NYSE_val-1
        g_price = price_val/prev_price_val-1
        list_NYSE.append(g_NYSE)
        list_price.append(g_price)
    prev_NYSE_val = NYSE_val
    prev_price_val = price_val
    i+=1
LR = linear_model.LinearRegression()

LR.fit(np.array(list_NYSE).reshape(-1,1),np.array(list_price).reshape(-1,1))
coef = LR.coef_
intercept = LR.intercept_

'''
print(coef)
[[0.24582618]]
print(intercept)
[0.00244132]
'''

'''*************************************
2B. Cost of Debt Formula
'''
print('existing rating')
#Existing Ratios
#the ratio below may not match 100% the fields chosen in 3A due to different data sources
#in real setting, ensure the fields are exactly the same. However for demo purpose, we use prox
r1= record_db['liabilities'] / record_db['assets']
r2=record_db['ebitda'] / record_db['assets']
r3= ( record_db['gp'] + record_db['intexp'] ) / record_db['assets']  
r4= ( record_db['netinc'] + record_db['depamor'] ) / record_db['liabilities']  
r5= ( record_db['gp'] ) / record_db['assets']  
r6= ( record_db['gp'] ) / record_db['revenue']  
r7= ( record_db['equity'] - record_db['bvps']*record_db['shareswa'] ) / record_db['assets']  
r8= ( record_db['netinc'] + record_db['depamor'] ) / record_db['liabilities']  
r9= math.log(record_db['assets'])
r10= ( record_db['gp'] + record_db['intexp'] ) / record_db['revenue']  
r11= record_db['opex'] / record_db['liabilitiesc']      
r12= ( record_db['ebitda'] ) / record_db['assets']
r13= record_db['workingcapital'] / record_db['assets']      
r14= ( record_db['ebitda'] ) /record_db['revenue']
r15= ( record_db['assetsc'] - record_db['inventory'] -record_db['receivables']) / record_db['liabilitiesc'] 
r16= record_db['ebitda'] / record_db['assets']      
r17= ( record_db['assetsc'] - record_db['inventory'] ) / record_db['liabilitiesc']  
r18= record_db['cor'] / record_db['revenue']
r19= record_db['revenue'] / record_db['liabilitiesc']

#do a cross check of 
X_test = [r1[0],r2[0],r3[0],r4[0],r5[0],r6[0],r7[0],r8[0],r9,r10[0],r11[0],r12[0],r13[0],r14[0],r15[0],r16[0],r17[0],r18[0],r19[0]]
f_logreg=open('log_reg.pkl',"rb")
logreg = pickle.load( f_logreg)

f_logreg_sc=open('logreg_scaler.pkl',"rb")
logreg_sc = pickle.load( f_logreg_sc)

X_array = np.asarray(X_test)
X_array = X_array.reshape(-1,1).transpose()
X_array = logreg_sc.transform(X_array)
default_risk_existing = logreg.predict_proba(X_array)[0][1]

print(default_risk_existing)


'''*************************************
2C. Calculate Drivers
'''
#Build linear regression
is_sgna_model= True
LR_cor = linear_model.LinearRegression()
LR_sgna = linear_model.LinearRegression()
LR_opex = linear_model.LinearRegression()
LR_ppnenet = linear_model.LinearRegression()
LR_payables = linear_model.LinearRegression()
LR_receivables = linear_model.LinearRegression()
LR_inventory = linear_model.LinearRegression()

LR_cor.fit(df_fs_combine[['revenue_chg']],df_fs_combine['cor_chg'])
try:
    LR_sgna.fit(df_fs_combine[['revenue_chg']],df_fs_combine['sgna_chg'])
except Exception:
    is_sgna_model = False
LR_opex.fit(df_fs_combine[['revenue_chg']],df_fs_combine['opex_chg'])
LR_ppnenet.fit(df_fs_combine[['revenue_chg']],df_fs_combine['ppnenet_chg'])
LR_receivables.fit(df_fs_combine[['revenue_chg']],df_fs_combine['receivables_chg'])
LR_payables.fit(df_fs_combine[['cor_chg']],df_fs_combine['payables_chg'])
LR_inventory.fit(df_fs_combine[['cor_chg']],df_fs_combine['inventory_chg'])

coef_cor = LR_cor.coef_
if is_sgna_model:
    coef_sgna = LR_sgna.coef_
coef_opex = LR_opex.coef_
coef_ppnenet = LR_ppnenet.coef_
intercept_cor = LR_cor.intercept_
if is_sgna_model:
    intercept_sgna = LR_sgna.intercept_
intercept_opex = LR_opex.intercept_
intercept_ppnenet = LR_ppnenet.intercept_
print('cor '+ str(coef_cor) + ' ' + str(intercept_cor))
print('opex '+ str(coef_opex) + ' ' + str(intercept_opex))
if is_sgna_model:
    print('sgna '+ str(coef_sgna) + ' ' + str(intercept_sgna))
print('ppnenet '+ str(coef_ppnenet) + ' ' + str(intercept_ppnenet))

LR_opex.fit(df_fs_combine[['cor_chg']],df_fs_combine['opex_chg'])
coef_opex = LR_opex.coef_
intercept_opex = LR_opex.intercept_
print('opex '+ str(coef_opex) + ' ' + str(intercept_opex))

coef_receivables = LR_receivables.coef_
intercept_receivables = LR_receivables.intercept_
print('receivables '+ str(coef_receivables) + ' ' + str(intercept_receivables))
coef_payables = LR_payables.coef_
intercept_payables = LR_payables.intercept_
print('payables '+ str(coef_payables) + ' ' + str(intercept_payables))

coef_inventory = LR_inventory.coef_
intercept_inventory = LR_inventory.intercept_
print('inventory '+ str(coef_inventory) + ' ' + str(intercept_inventory))

#correlation matrix
file_corr_out = 'corr.csv'
df_corr = df_fs_combine[['revenue_chg','cor_chg','opex_chg','ppnenet_chg','receivables_chg','payables_chg','inventory_chg']] * 100
df_corr = df_corr.astype(float)
corr_m = df_corr.corr()
sns.heatmap(corr_m)
corr_m.to_csv(file_corr_out)
plt.show()

'''*************************************
3. Projection
'''
print('optimization...')
#simulations
record_db_f = record_db

#Projection
price = record_db_f['price']
low_price = price * 0.5
optimal_WACC = 1
optimal_new_debt_pct = 0
optimal_pricing_offering = 0
f_log = open('log.txt','w+')
debt_pct_range = np.arange(0,1,0.01)
price_offering_range = np.arange(low_price[0],price[0],0.1)

def cal_F_WACC(record_db_f, logreg, logreg_sc, new_debt_pct,price_offering,levered_beta,sales_growth,coefs,r_free):
    new_equity_pct = 1- new_debt_pct
    #levered_beta = coef = LR.coef_[0]
    #levered_beta = 0.24582618
    #sales_growth = 0.10
    
    #coef_sales_ppenet = -0.00012991
    #coef_sales_cor	= 0.69804978
    #coef_cor_opex = 0.35883998
    
    coef_sales_ppenet = coefs[0]
    coef_sales_cor	= coefs[1]
    coef_cor_opex = coefs[2]

    #r_free = 0.00244132
    
    sales_ppenet = record_db_f['revenue'][0]/record_db_f['ppnenet'][0]
    d_e_ratio = record_db_f['debt'][0]/record_db_f['equity'][0]
    tax_pct = record_db_f['taxexp'][0]/ (record_db_f['ebit'][0]-record_db_f['intexp'][0])
    unlevered_beta = levered_beta/(1+((1-tax_pct)*d_e_ratio))
    F_sales = record_db_f['revenue'][0] * (1+sales_growth)
    F_net_ppenet = F_sales / sales_ppenet * (1+ coef_sales_ppenet)
    
    ap_days = record_db_f['payables'][0]/record_db_f['cor'][0]*365
    ar_days = record_db_f['receivables'][0]/record_db_f['revenue'][0]*365
    inventory_days = record_db_f['inventory'][0]/record_db_f['cor'][0]*365
    total_cash_cycle = ar_days + inventory_days - ap_days
    cash_sales = total_cash_cycle/(inventory_days+ar_days)*record_db_f['cor'][0]/record_db_f['revenue'][0]
    cash_opex = record_db_f['opex'][0]/2/record_db_f['revenue'][0]
    wc_sales = (cash_sales+cash_opex)*F_sales
    total_capital_needed = F_net_ppenet+wc_sales+record_db_f['inventory'][0]+record_db_f['intangibles'][0]
    debt_repayment = record_db_f['ncfdebt'][0]
    dividend_payment = record_db_f['ncfdiv'][0]
    existing_capital = record_db_f['debt'][0] - debt_repayment + record_db_f['equity'][0]-dividend_payment
    if total_capital_needed > existing_capital:
        new_capital_required = total_capital_needed - existing_capital
    else:
        new_capital_required = 0    
    cor_growth = (1 + sales_growth * coef_sales_cor) - 1
    F_cor = record_db_f['cor'][0]*(1+cor_growth)
    F_gross_profit = F_sales - F_cor
    F_sgna = 0
    opex_growth = (1+cor_growth*coef_cor_opex)-1
    F_opex = record_db_f['opex'][0]*(1+opex_growth)
    F_ebitda = F_sales - F_cor - F_sgna
    F_ebit = F_sales - F_cor - F_sgna - F_opex
    F_WC = record_db_f['inventory'][0] + record_db_f['receivables'][0]-record_db_f['payables'][0]
    F_new_equity = new_capital_required * new_equity_pct
    F_new_debt = new_capital_required * new_debt_pct
    F_equity = record_db_f['equity'][0] + new_capital_required * new_equity_pct
    F_debt = record_db_f['debt'][0] + new_capital_required * new_debt_pct
    F_asset = (F_debt+F_equity)
    
    #New Ratios for default risk
    r1= F_debt / F_asset
    r2= F_ebitda / F_asset
    r3= ( record_db_f['gp'][0] + record_db_f['intexp'][0] ) / F_asset
    r4= ( record_db_f['netinc'][0] + record_db_f['depamor'][0] ) / record_db_f['liabilities'][0]
    r5= record_db_f['gp'][0]/F_asset
    r6= record_db_f['gp'][0]/record_db_f['revenue'][0]
    r7=( record_db_f['equity'][0] - record_db_f['bvps'][0]*record_db_f['shareswa'][0] ) / record_db_f['assets'][0]  
    r8= ( record_db_f['netinc'][0] + record_db_f['depamor'][0] ) / record_db_f['liabilities'][0]
    r9= math.log(F_asset)
    r10=(record_db_f['gp'][0] + record_db_f['intexp'][0] ) / record_db_f['revenue'][0]
    r11=F_opex / record_db_f['liabilitiesc'][0]
    r12=F_ebitda /record_db_f['assetsc'][0]
    r13=F_WC/record_db_f['assets'][0]
    r14=F_ebitda / F_sales
    r15=( record_db_f['assetsc'][0] - record_db_f['inventory'][0] - record_db_f['receivables'][0] ) / record_db_f['liabilitiesc'][0]
    r16=F_ebitda / F_asset
    r17=( record_db_f['assetsc'][0] - record_db_f['inventory'][0]) / record_db_f['liabilitiesc'][0]
    r18= F_cor / F_sales
    r19=F_sales / F_debt
    
    F_X_test = [r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19]
    F_X_array = np.asarray(F_X_test)
    F_X_array = F_X_array.reshape(-1,1).transpose()
    F_X_array = logreg_sc.transform(F_X_array)
    F_Y_pred_logreg = logreg.predict_proba(F_X_array)
    F_default_risk = F_Y_pred_logreg[0][1]
    
    r_debt = F_default_risk + r_free
    F_intexp = r_debt * F_debt
    F_ebt = F_ebit - F_intexp
    if F_ebt > 0:
        F_taxexp = tax_pct * F_ebt
    else:
        F_taxexp = 0
    F_earnings = F_ebt - F_taxexp
    F_D_E = F_debt / F_equity
    r_equity = unlevered_beta*(1+((1-tax_pct)*F_D_E))+r_free
    
    #objective
    F_WACC = F_debt/ F_asset * (1-tax_pct) * r_debt + F_equity/F_asset * r_equity
    
    #equity offering constraints --- not bounding
    price_offering = record_db_f['price'][0]
    unit_offering = int(F_new_equity / price_offering)
    F_eps = F_earnings / (unit_offering+record_db_f['shareswa'][0])
    equity_growth = F_equity / record_db_f['equity'][0]-1
    eps_growth = abs(F_eps/ (record_db_f['netinc'][0]/record_db_f['shareswa'][0])-1)
    c_eq_1 = equity_growth <= 0.1
    c_eq_2 = eps_growth <= 0.3
    return F_WACC, F_default_risk,(c_eq_1,c_eq_2)

for new_debt_pct in debt_pct_range:
    for price_offering in price_offering_range:
        r_free = 0.00244132
        levered_beta = 0.24582618
        sales_growth = 0.10
        coef_sales_ppenet = -0.00012991
        coef_sales_cor	= 0.69804978
        coef_cor_opex = 0.35883998
        coefs = (coef_sales_ppenet,coef_sales_cor,coef_cor_opex)
        F_WACC, F_default_risk,conditions = cal_F_WACC(record_db_f,logreg,logreg_sc, new_debt_pct, price_offering,levered_beta,sales_growth,coefs,r_free)
        '''****************************************
        4. Calculate WACC
        '''
        #update WACC
        obj = F_WACC < optimal_WACC and F_default_risk/default_risk_existing-1<=0.75
        if obj:
            optimal_WACC = F_WACC
            optimal_new_debt_pct = new_debt_pct
            optimal_price_offering = price_offering
            print('update at ' + str(F_WACC) + ' price ' + str(price_offering)+ ' debt ' + str(optimal_new_debt_pct))
            f_log.write(str(F_WACC) + '\t'+str(price_offering) + '\t'+str(new_debt_pct) + '\n')

'''
cor [[0.69804978]] [0.0093667]
opex [[0.2233317]] [-0.01642346]
ppnenet [[-0.00012991]] [-0.01224134]
opex [[0.35883998]] [-0.01927296]

update at 0.1067572997698917 price 84.11 debt 0.86
'''

'''
#sudo apt-get install coinor-clp
#conda install -c conda-forge pulp 

import pulp

model = pulp.LpProblem("WACC minimising problem", pulp.LpMinimize)
new_debt_pct = pulp.LpVariable('debt_pct', lowBound=0, upBound=1, cat='Continuous')
price_offering = pulp.LpVariable('price_offering', lowBound=0, cat='Continuous')
'''
