#!/usr/bin/env python3
# -*- coding: utf-8 -*-
QUANDLKEY = '<Enter your Quandl APT key here>'
"""
Created on Mon Sep 24 18:14:11 2018

@author: jeff

"""
'''************************
Step1
5A_S1_1a_instutional_holdering:
Deliver the institutional holdings with regards to movement in various market segment(scale) - so that we can run clustering model on this dataset
'''

'''************************
1a) Load Data
'''
#import relevant libraries
import quandl
from datetime import date,timedelta
import pandas as pd
import os

#load tickers universe and description field (scale)
print('load ticker universe')

df_tkr = pd.read_csv('industry_tickers_list.csv')


dict_scale_tkr = {}
for index, row in df_tkr.iterrows():
    scale = row['scalemarketcap']
    tkr = row['ticker']
    dict_scale_tkr[tkr] = [scale]

start_d =date(2018,1,1)
end_d =date(2018,1,5)

#loop through investors
quandl.ApiConfig.api_key = QUANDLKEY
#comment this out if you prefer the longer list
f_name = open('investors_select.txt','r')
#use this if you prefe the full list
#f_name = open('investors.txt','r')

investorNameList = f_name.readlines()

st_yr = 2013
end_yr = 2019
qtr_mmdd_list= ['-03-31','-06-30','-09-30','-12-31']
prev_investor = ""
prev_data_df = pd.DataFrame()
current_file_dir = os.path.dirname(__file__)
delta = timedelta(days=1)
print('prep investor movement')

for investor in investorNameList:
    investor = investor.rstrip('\n')
    print(investor)
    curr_d = start_d
    investor_df = pd.DataFrame()
    data_df = pd.DataFrame()
    prev_investor_df = pd.DataFrame()
    prev_investor = ''
    #calculate the change in position by ticker on Quarter-to-quarter basis
    for yr_num in range(st_yr,end_yr):
        yr = str(yr_num)
        for mmdd in qtr_mmdd_list:
            dte_str = yr + mmdd
            print(dte_str)
            try:
                data_df = quandl.get_table("SHARADAR/SF3", paginate=True,investorname=investor,calendardate=dte_str)
            except Exception:
                print('no data')
                next
            if (len(data_df)>0 and len(prev_data_df)>0):
                df_combined = data_df.merge(prev_data_df, on='ticker')
                #fld_y is prev, fld_x is current
                df_combined['units_chg'] = df_combined['units_x'] - df_combined['units_y']
                df_combined['price_chg'] = df_combined['price_x'] - df_combined['price_y']
                if len(investor_df)==0:
                    investor_df = df_combined
                else:
                    investor_df = investor_df.append(df_combined)
            prev_data_df = data_df
            
    #qualify investor's activities
    print('classify investor decision')
    investor_df['action'] =''
    investor_df['scale'] = ''
    for index, row in investor_df.iterrows():
        try:
            this_scale = dict_scale_tkr[row['ticker']][0]
        except Exception:
            continue
        #is_scale = (this_scale in list_scale)
        if row['units_chg'] < 0:
            if row['units_x'] == 0:
                investor_df.at[index,'action']='SELL-ALL'
            else:
                investor_df.at[index,'action']='SELL-PARTIAL'
        elif row['units_chg'] > 0:
            if row['units_y'] == 0:
                investor_df.at[index,'action']='BUY-NEW'
            else:
                investor_df.at[index,'action']='BUY-MORE'
        else:
            investor_df.at[index,'action']='HOLD'
        investor_df.at[index,'scale']=this_scale
    #output the ticker’s activities of the investor
    output_path = os.path.join(current_file_dir,'data',investor+'.csv')
    investor_df.to_csv(output_path)
