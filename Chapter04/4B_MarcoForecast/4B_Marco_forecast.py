#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 16:15:03 2018

@author: jeff
"""
from pyquery import PyQuery
import pandas as pd
import quandl
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score

cnt = 0
fname = 'tmp.csv'
i_col_name = 'yyyy-mm-dd'
#df_mean = pd.DataFrame(columns = ['yymm','mean'])

for year in range(2013,2018):
    for month in range(1, 13):
        mth_str = '{:02}'.format(month)
        yr_str = str(year)
        print(yr_str+mth_str)
        html = "./weather/"+yr_str+"-"+mth_str+".html"
        pq = PyQuery(filename = html)
        tag = pq('div#results_area') # or     tag = pq('div.class')
        f = open(fname,'w+')
        f.write(tag.text())
        f.close()
        tmp_df = pd.read_csv(fname,header=1)
        tmp_df['yyyy-mm-dd'] = (yr_str+'-'+ mth_str+'-01')
        tmp_df[' MeanAvgTemperature'] = pd.to_numeric(tmp_df[' MeanAvgTemperature'],errors='coerce')
        mean_tmp = tmp_df[' MeanAvgTemperature'].mean()
        if cnt ==0:
            df_all = tmp_df
        else:
            df_all = df_all.append(tmp_df)
        cnt+=1
df_all[i_col_name ]= pd.to_datetime(df_all[i_col_name ])
df_mean= df_all.groupby([i_col_name]).mean()
df_all.to_csv('weather_df.csv')
df_mean.to_csv('weather_mean.csv')
#from pandas.tseries.offsets import *
cal_LIND = quandl.get("FRED/CASLIND", authtoken="[quandl id]")
cal_ele = quandl.get(["EIA/ELEC_SALES_CA_RES_M","EIA/ELEC_SALES_CA_IND_M"], authtoken="<Enter your Quandl APT key here>")

#update the index date to begin of month (in fact all index should be referring to end of month)
cal_ele['mth_begin'] = cal_ele.index
#change the column to begin of month
for index, row in cal_ele.iterrows():
    cal_ele.set_value(index,'mth_begin', pd.datetime(row['mth_begin'].year,row['mth_begin'].month,1))
cal_ele= cal_ele.reset_index(drop = True)
cal_ele.index = cal_ele['mth_begin']
cal_ele = cal_ele.drop(['mth_begin'],axis=1)

df_temp= cal_LIND.merge(df_mean, left_index=True,right_index=True)
df_marco = df_temp.merge(cal_ele,left_index = True, right_index = True)


reg_retail = linear_model.LinearRegression()
reg_retail.fit(df_marco[[' MeanAvgTemperature']], df_marco['EIA/ELEC_SALES_CA_RES_M - Value'])
reg_retail.coef_
reg_retail_pred = reg_retail.predict(df_marco[[' MeanAvgTemperature']])
error_retail = r2_score(df_marco['EIA/ELEC_SALES_CA_RES_M - Value'], reg_retail_pred)

reg_ind = linear_model.LinearRegression()
reg_ind.fit(df_marco[[' MeanAvgTemperature']], df_marco['EIA/ELEC_SALES_CA_IND_M - Value'])
reg_ind.coef_
reg_ind_pred = reg_ind.predict(df_marco[[' MeanAvgTemperature']])
error_ind = r2_score(df_marco['EIA/ELEC_SALES_CA_IND_M - Value'], reg_ind_pred)

df_marco.to_csv('marco_output.csv')

#df_marco[' MeanAvgTemperature'].plot(color='blue',grid=True,label='temperature',title='temperature vs industrial electricity')
#df_marco['EIA/ELEC_SALES_CA_IND_M - Value'].plot(color='red',grid=True,label='industrial electricity')
#plt.legend()
fig, plt1 = plt.subplots()
plt1.set_xlabel('year-mth')
plt1.set_ylabel('temperature', color='blue')
plt1.plot(df_marco.index, df_marco[' MeanAvgTemperature'], color='blue')
plt2 = plt.twinx()
plt2.set_ylabel('Industrial Electricity', color='red')
plt2.plot(df_marco.index, df_marco['EIA/ELEC_SALES_CA_IND_M - Value'], color='red')
plt.show()
plt.close()

'''
error_ind
Out[3]: 0.7466844036675353

error_retail
Out[4]: 0.37044441701064523
'''
