#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 23:59:17 2018
@author: jeff
As a CFA, the coding here is for educational purpose - it is not intended to be use as any form of investment advisor.
Please use the code esp on the choice of asset with a grain of salt!!!!
"""
'''*************************************
#1. Import libraries and key varable values
'''
QUANDLKEY = '<ENTER YOUR QUANDLKEY HERE>'

import quandl
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
import sqlite3
#not needed when using database
import pickle

#API Key
quandl.ApiConfig.api_key = QUANDLKEY

#dates variables for all the download
str_dte = '2008-01-01'
end_dte = '2018-7-31'
date_dict = {'gte':str_dte, 'lte':end_dte}

#db file
db_path = 'ETF_Para.db'


'''*************************************
#2. Define function to download data for each of the asset
'''
def download_tkr(tkr,date_dict):
    temp_X=quandl.get_table('SHARADAR/SFP',date=date_dict,ticker=tkr)
       
    temp_X['Mid'] = (temp_X['high'] + temp_X['low'])/2
    temp_X = temp_X.sort_index(ascending = False)
    temp_X.index = temp_X['date']
    temp_X = temp_X.dropna()
    temp_X.to_csv('~/temp_X.csv')
    X_diff = temp_X['Mid'].diff()
    X_diff=X_diff.dropna()
    X_diff.index = temp_X[:-1].index
    X_diff.to_csv('~/X_diff.csv')
    temp_X['Mid_chg'] =(X_diff+0.0)
    temp_X.to_csv('~/temp_X1.csv')
    fld = tkr+'_Mid_chg'
    temp_X[fld] = temp_X['Mid_chg']/ temp_X['Mid']+1
    X = temp_X[fld]
    return X

'''*************************************
#3. Market Return
'''

#day count
day_cnt = 360

ind_tkr = "VT"
X_pd = download_tkr(ind_tkr,date_dict)
X_pd_prod = X_pd.prod()
day_sum = len(X_pd)
r_m = X_pd_prod**(day_cnt/day_sum)-1
var_m = np.var(X_pd)

'''*************************************
#4. Risk Free Rate
'''
#risk free rate
rf_pd = quandl.get("FRED/DTB3",start_date=str_dte, end_date=end_dte)
rf_pd['year'] = rf_pd.index.year
rf_gp = rf_pd.groupby(['year']).count()
rf_gp['Day_cnt'] = rf_gp['Value']
rf_gp['year'] = rf_gp.index
rf_gp.reset_index(drop=True,inplace=True)
rf_gp = rf_gp.drop('Value',axis = 1)
rf_pd_i =  rf_pd.merge(rf_gp,on='year')
#daily return at daily interest rate
rf_pd_i['daily_interest'] = (1+ rf_pd_i['Value']/100/day_cnt)
rf_pd_i_gp = rf_pd_i['daily_interest'].prod()
day_sum = len(rf_pd)
r_f = rf_pd_i_gp**(day_cnt/day_sum)-1
#0.003985863616917662

# override return of market
#TODO: fix the negative market risk premium problem
r_m = max(r_m,r_f+0.00001)

'''*************************************
#5. Asset Return and parameters
'''
para_dict = {}
#list of stocks for selection in the active portfolio
list_tkr = ['DWX','TIPX','FLRN','CBND','SJNK','SRLN','CJNK','DWFI','EMTL','STOT','TOTL','DIA','SMEZ','XITK','GLDM','GLD','XKFS','XKII','XKST','GLDW','SYE','SYG','SYV','LOWC','ZCAN','XINA','EFAX','QEFA','EEMX','QEMM','ZDEU','ZHOK','ZJPN','ZGBR','QUS','QWLD','OOO','LGLV','ONEV','ONEO','ONEY','SPSM','SMLV','MMTM','VLU','SPY','SPYX','SPYD','SPYB','WDIV','XWEB','MDY','NANR','XTH','SHE','GAL','INKM','RLY','ULST','BIL','CWB','EBND','JNK','ITE','IBND','BWX','SPTL','MBG','BWZ','IPE','WIP','RWO','RWX','RWR','FEZ','DGT','XNTK','CWI','ACIM','TFI','SHM','HYMB','SPAB','SPDW','SPEM','SPIB','SPLG','SPLB','SPMD','SPSB','SPTS','SPTM','MDYG','MDYV','SPYG','SPYV','SLY','SLYG','SLYV','KBE','KCE','GII','KIE','KRE','XAR','XBI','GXC','SDY','GMF','EDIV','EWX','GNR','XHE','XHS','XHB','GWX','XME','XES','XOP','XPH','XRT','XSD','XSW','XTL','XTN','FEU','PSK']
master_pd = pd.DataFrame()

#connect to the databases and reset it everytime with drop indicator
conn = sqlite3.connect(db_path)
c = conn.cursor()
drop = True
if drop== True:
    sqlstr = "drop table ETF_para"
    try:
        output = c.execute(sqlstr)
    except Exception:
        print('non exists')
sqlstr = "CREATE TABLE IF NOT EXISTS ETF_para(  TICKER TEXT PRIMARY KEY,  beta REAL,  alpha REAL NULL,  var_err REAL NULL)"
c.execute(sqlstr)

#write out the risk free and market parameters
f = open('r.txt','w+')
f.write(str(r_f)+','+str(r_m)+','+str(var_m))
f.close()

#loop through the tickers
for tkr in list_tkr:
    print(tkr)
    #calculate the CAPM:
    #download data for the ticker
    Y_tkr= download_tkr(tkr,date_dict)
    
    #make sure the ticket we select has market data
    if len(Y_tkr)>0:
        #linear regression
        LR = linear_model.LinearRegression()
        combine_pd = pd.concat([X_pd,Y_tkr],axis=1, join = "inner")
        combine_pd= combine_pd.dropna()
        if len(master_pd) == 0:
            master_pd = combine_pd
        Y_actual = combine_pd[combine_pd.columns[1]].values.reshape(-1, 1)
        X = combine_pd[combine_pd.columns[0]].values.reshape(-1, 1)
        LR.fit(X,Y_actual)
        Y_pred = LR.predict(X)
        #obatin the result and write out the parameters
        r2_sc = r2_score(Y_actual,Y_pred)
        coef = LR.coef_[0]
        intercept = LR.intercept_
        variables_dict = {}
        variables_dict['beta']=str(coef[0])
        variables_dict['alpha']=str(intercept[0]-r_f)
        variables_dict['var_err']= str(r2_sc)
        para_dict[tkr] = variables_dict
        master_pd = pd.concat([master_pd,Y_tkr],axis=1, join = "outer")
        item = [(tkr,coef[0],intercept[0]-r_f,r2_sc)]
        sqlstr = 'insert into ETF_para(TICKER,beta,alpha,var_err) VALUES (?,?,?,?)'
        c.executemany(sqlstr, item)

#closing out and output the results
conn.commit()
c.close()

#from this line onwards - not needed as we outputted to a database
print(para_dict)
f=open('para.txt','w+')
f.write('tkr,'+'coef,'+'intercept,'+'r2\n')
f.write(str(para_dict))
f.close()

f=open('para_dict.pkl','wb+')
pickle.dump(para_dict,f)

master_pd.to_csv('master_data.csv')