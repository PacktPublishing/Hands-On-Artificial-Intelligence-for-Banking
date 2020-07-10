#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 00:58:34 2018

@author: jeff
"""
'''*************************************
#1. Import libraries and key varable values
'''
import os
import quandl
import pandas as pd
import numpy as np
import keras
from PIL import Image

#folder path
folder_path = os.path.dirname(__file__)

#date range for full dataset
str_dte = '2003-01-01'
end_dte = '2018-7-31'
date_dict = {'gte':str_dte, 'lte':end_dte}
#Dates for back-testing
start_dte = '2015-1-1'
#Create list of dates
datelist = pd.date_range(start_dte, periods=365*2).tolist()

#API key for quandl
quandl.ApiConfig.api_key = '[quandl id]'

#Parameters for the image generation
col_num_mid = 10
col_num_dte = 9
pixel_size = 100
window_size = 60
pred_window_size = 1

#model path
model_path = "model2_2DCov.h5"
model = keras.models.load_model(model_path)
#number of channel for the image
num_channel=1

#strategies parameters
curr_pnl = 10000
curr_pnl_0=curr_pnl
curr_pnl_1=curr_pnl
curr_pnl_2=curr_pnl
quant_trans_0 = 0
quant_trans_1 = 0
quant_trans_2 = 0
min_pnl = 0.0005
trading_cost = 0
trade_limit = 0.5

'''*************************************
#2. Define functions
'''

#input_X is a series of price
#output_X is a series of price expressed in pixel
def rescale(input_X, pixel, min_x,max_x):
    unit = (max_x - min_x)/pixel
    output_X = round((input_X-min_x)/unit,0)
    return output_X,unit

'''*************************************
#3. Running the test
'''
#Get the data
tkr = 'VTV'
df =quandl.get_table('SHARADAR/SFP',date=date_dict,ticker=tkr)
df = df.sort_values(by=['date'])
df=df.reset_index(drop=True)

#write header for the log of the strategy back-testing
f = open('log.txt','w+')
f.write('strategy\tBuySell\t' + 'dte' +'\t'+ 'cost' +'\t'+ 'T+1_actual' +'\t'+ 'T+1_pred'+'\t'+ 'Quantity'+'\t'+ 'PnL'+'\n')

#loop through the dates
for pred_dte in datelist:
    df_i = df.index[df['date']==pred_dte]
    
    #make sure both start and end dates are valid
    if df_i.empty:
        print('no data')
        continue
   
    df_i = df_i[0]
    print(pred_dte)
    df_start = df_i-(window_size) #starts at zero
   
    if df_start < 0: #in case the date inputted is not valid
        print('later date')
        continue
   
    #prepare the input data
    df['mid'] = (df['high'] + df['low'])/2
    df_plot = df.iloc[df_start:df_i,:]
    min_p = min(df_plot['mid'])
    max_p = max(df_plot['mid'])
    output_pixel,unit = rescale(df_plot['mid'],pixel_size,min_p,max_p)

    #if no trend, then drop this data point    
    if min_p ==max_p:
        print('no trend')
        continue
    
    #stack up for a numpy for Image Recognition
    #print the historical data
    img_ar = np.zeros((1,pixel_size,window_size,num_channel))
    img_display = np.zeros((pixel_size,window_size,num_channel))
    k=0
    pix_p=0
    for pix in output_pixel:
        y_pos = int(pix)-1
        img_ar[0][y_pos][k][num_channel-1] = 255
        img_display[y_pos][k][num_channel-1] = 255
        pix_p=y_pos
        k+=1
    img_row = img_ar/255
    
    last_actual_p = pix_p * unit + min_p
    
    #make prediction
    pred_y = model.predict(img_row)
    max_y_val = max(pred_y[0])
    pred_y_img = np.zeros((pixel_size,1))
    
    #Obtain predicted price
    pred_pixel = 0
    expected_p = 0
    #calculate expected values
    for i in range(pixel_size):
        expected_p += pred_y_img[i,0] * i
        if pred_y[0,i] == max_y_val:
            pred_y_img[i,0] = 255
            pred_pixel = i
    pred_p = pred_pixel * unit + min_p
    print('cost at ' + str(last_actual_p))
    print('predict p be ' + str(pred_p) + ' and probability of ' + str(max_y_val))
    pred_exp_p = expected_p * unit + min_p
    print('expected predict p be ' + str(pred_exp_p))
    y_actual_p = df.iloc[df_i+1,:]['mid']
    print('actual p be '+str(y_actual_p))
    
    #Strategy Back-Testing
    #Benchmark - Strategy 0 - buy and hold
    if quant_trans_0 == 0:
        quant_trans_0 = curr_pnl/y_actual_p
        pnl = 0-trading_cost
    else:
        pnl = (y_actual_p/last_actual_p-1) * quant_trans_0
    curr_pnl_0 += pnl
    f.write('B0\tNA\t' + str(pred_dte) +'\t'+ str(last_actual_p) +'\t'+ str(y_actual_p) +'\t'+ str(y_actual_p)+'\t'+ str(1)+'\t'+ str(last_actual_p-y_actual_p)+'\n')

    #Testing of strategy1
    order_type = ""
    quant_trans_1 = int(curr_pnl_1/last_actual_p*0.5)
    if abs(pred_exp_p/last_actual_p-1)>min_pnl:
        if pred_exp_p>last_actual_p:
            #buy one now / long one unit
            #stock_unit_1+=quant_trans_1
            pnl = (y_actual_p-last_actual_p)  * quant_trans_1-trading_cost
            order_type = "B"
            curr_pnl_1 += pnl
            f.write('S1\tBuy\t' + str(pred_dte) +'\t'+ str(last_actual_p) +'\t'+ str(y_actual_p) +'\t'+ str(pred_exp_p)+'\t'+ str(quant_trans_1)+'\t'+ str(y_actual_p-last_actual_p)+'\n')
        elif pred_exp_p<last_actual_p:
            #sell one now / short one unit
            #stock_unit_1-=quant_trans_1
            pnl = (last_actual_p-y_actual_p) * quant_trans_1-trading_cost
            order_type = "S"
            curr_pnl_1 += pnl
            f.write('S1\tSell\t' + str(pred_dte) +'\t'+ str(last_actual_p) +'\t'+ str(y_actual_p) +'\t'+ str(pred_exp_p)+'\t'+ str(quant_trans_1)+'\t'+ str(last_actual_p-y_actual_p)+'\n')
        else: #no trade
            if order_type == "B":
                pnl = (y_actual_p-last_actual_p)  * quant_trans_1
            else:
                pnl = (last_actual_p-y_actual_p) * quant_trans_1
            curr_pnl_1 += pnl
        
    #Testing of strategy2
    if max_y_val > 0.99 and abs(pred_p/last_actual_p-1)>min_pnl:
        quant_trans_2 = int(curr_pnl_2/last_actual_p*0.5)
        if pred_p>last_actual_p:
            #buy one now / long one unit
            #stock_unit_2+=quant_trans_2
            order_type = "B"
            curr_pnl_2 += (y_actual_p-last_actual_p) * quant_trans_2-trading_cost
            f.write('S2\tBuy\t' + str(pred_dte) +'\t'+ str(last_actual_p) +'\t'+ str(y_actual_p) +'\t'+ str(pred_p) +'\t'+str(quant_trans_2)+'\t'+ str(y_actual_p-last_actual_p)+'\n')

        elif pred_p<last_actual_p:
            #sell one now / short one unit
            #stock_unit_2-=quant_trans_2
            order_type = "S"
            curr_pnl_2 += (last_actual_p-y_actual_p) * quant_trans_2-trading_cost
            f.write('S2\tSell\t' + str(pred_dte) +'\t'+ str(last_actual_p) +'\t'+ str(y_actual_p) +'\t'+ str(pred_p)+'\t'+ str(quant_trans_2)+'\t'+ str(last_actual_p-y_actual_p)+'\n')
        else: #no trade
            if order_type == "B":
                pnl = (y_actual_p-last_actual_p)  * quant_trans_2
            else:
                pnl = (last_actual_p-y_actual_p) * quant_trans_2
            curr_pnl_2 += pnl
#print the final result of the strategies
print(curr_pnl_0)            
print(curr_pnl_1)
print(curr_pnl_2)
f.close()

'''
export CUDA_VISIBLE_DEVICES=''
tensorboard --logdir AI_Finance_book/6B_TrendFollowings/Graph/ --host localhost --port 6006
'''
