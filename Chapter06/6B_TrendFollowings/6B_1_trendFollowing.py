#!/usr/bin/env python3
# -*- coding: utf-8 -*-
QUANDLKEY = '<Enter your Quandl APT key here>'
"""
Created on Thu Oct 25 23:19:44 2018

@author: jeff
"""
'''*************************************
#1. Import libraries and key varable values

'''
import quandl
import plotly
import plotly.graph_objs as go
import numpy as np

from datetime import datetime
try:
    import Image
except ImportError:
    from PIL import Image
import os
import h5py

#dates parameters
str_dte = '2003-01-01'
end_dte = '2018-7-31'
date_dict = {'gte':str_dte, 'lte':end_dte}

#quandl setting
quandl.ApiConfig.api_key = QUANDLKEY
col_num_mid = 10
col_num_dte = 9

#parameters for the image generation
pixel_size = 100
window_size = 60
pred_window_size = 1
num_channel = 1

#create path for the output dataset
folder_path = os.path.dirname(__file__)
data_X_dir = os.path.join(folder_path,'dataset')
data_Y_dir = os.path.join(folder_path,'dataset')

#ticker lists
#tkr_list = ['TIPX','HYMB','TFI','ULST','MBG','FLRN','SHM','STOT','SPTS','BIL','SPSB']
tkr_list = ['DWX','TIPX','FLRN','CBND','SJNK','SRLN','CJNK','DWFI','EMTL','STOT','TOTL','DIA','SMEZ','XITK','GLDM','GLD','XKFS','XKII','XKST','GLDW','SYE','SYG','SYV','LOWC','ZCAN','XINA','EFAX','QEFA','EEMX','QEMM','ZDEU','ZHOK','ZJPN','ZGBR','QUS','QWLD','OOO','LGLV','ONEV','ONEO','ONEY','SPSM','SMLV','MMTM','VLU','SPY','SPYX','SPYD','SPYB','WDIV','XWEB','MDY','NANR','XTH','SHE','GAL','INKM','RLY','ULST','BIL','CWB','EBND','JNK','ITE','IBND','BWX','SPTL','MBG','BWZ','IPE','WIP','RWO','RWX','RWR','FEZ','DGT','XNTK','CWI','ACIM','TFI','SHM','HYMB','SPAB','SPDW','SPEM','SPIB','SPLG','SPLB','SPMD','SPSB','SPTS','SPTM','MDYG','MDYV','SPYG','SPYV','SLY','SLYG','SLYV','KBE','KCE','GII','KIE','KRE','XAR','XBI','GXC','SDY','GMF','EDIV','EWX','GNR','XHE','XHS','XHB','GWX','XME','XES','XOP','XPH','XRT','XSD','XSW','XTL','XTN','FEU','PSK']
#generate png file for each of the input or now
img_output =False

#generate interactive plot to the ticket stock price or not
gen_plot = False
'''*************************************
#2. Define the function to rescale the stock price according to the min and max values

'''
#input_X is a series of price
#output_X is a series of price expressed in pixel
def rescale(input_X, pixel, min_x,max_x):
    unit = (max_x - min_x)/pixel
    output_X = round((input_X-min_x)/unit,0)
    return output_X,unit


'''*************************************
#3. Go through the tickers
'''
for tkr in tkr_list:
    print(tkr)
    
    #if the ticker has been downloaded, skip the ticket and go for the next one
    if os.path.exists(tkr+'6b1_completed.txt'):
        continue
    
    #download and create dataset
    df =quandl.get_table('SHARADAR/SFP',date=date_dict,ticker=tkr)
    #sort the date from ascending to descending...
    df = df.sort_values(by=['date'])
    df=df.reset_index(drop=True)
    
    #charting interactive chart for viewing the data
    if gen_plot == True:
        trace = go.Candlestick(x=df.date,
                               open=df.open,
                               high=df.high,
                               low=df.low,
                               close=df.close)
        data = [trace]

        plotly.offline.plot(data, filename=tkr+'simple_candlestick')

    #calculate mid price of the day
    df['mid'] = (df['high'] + df['low'])/2
    len_df = len(df)
    num_img = max(int(len_df-window_size-1),0)
    current_min_dte = df.date
    
    train_shape = (num_img, pixel_size, window_size,num_channel)
    label_shape = (num_img, pixel_size)
    
    #remove the file if there is one
    data_X_path = os.path.join(data_X_dir,tkr+'X_img.h5')
    try:
        os.remove(data_X_path)
    except OSError:
        pass
    h5f_X = h5py.File(data_X_path,'w')

    #remove the file if there is one
    data_Y_path = os.path.join(data_Y_dir,tkr+'Y_label.h5')
    try:
        os.remove(data_Y_path)
    except OSError:
        pass
    h5f_Y = h5py.File(data_Y_path,'w')
    
    #create dataset within the HDF5 file
    #now we create the dataset with a fixed size to fit all the data, it could also be create to fit fixed batches    
    h5f_X.create_dataset("X_img_ds", train_shape, np.float32)
    h5f_Y.create_dataset("Y_label_ds", label_shape, np.float32)
    
    #loop through the dates
    for i in range(num_img):
        img_ar = np.zeros((pixel_size,window_size,1))
        result_Y =np.zeros((pixel_size))
        df_plot = df.iloc[i:window_size+i,:]
        
        #create min and max values for the mid price plot within a given timeframe
        min_p = min(df_plot['mid'])
        max_p = max(df_plot['mid'])
        output_pixel,unit = rescale(df_plot['mid'],pixel_size,min_p,max_p)
        df_next = df.iloc[window_size+i+1,:]
        next_p = df_next['mid']
        next_p_val = max(round((min(next_p,max_p)-min_p)/unit,0),0)

        #in case of low liquidity ETF which has the same price, no graph be drawn
        if min_p ==max_p:
            continue

        k = 0
        #draw the dot on the x, y axis of the input image array
        for pix in output_pixel:
            img_ar[int(pix)-1][k][0] = 255
            k+=1
        
        #output the image for visualization
        if img_output:
            img = Image.fromarray(img_ar)
            if img.mode != 'RGB':
                new_img = img.convert('RGB')
            file_path = os.path.join(folder_path,'img/'+tkr+str(i)+'.png')
            new_img.save(file_path,"PNG")
        img_row = img_ar/255
                
        #draw the dot on the target image for training
        result_Y[int(next_p_val)-1] = 255
        result_Y_row=result_Y/255
        
        #stack up for a numpy for Image Recognition
        h5f_X["X_img_ds"][i, ...] = img_row
        h5f_Y["Y_label_ds"][i, ...] = result_Y_row
            
        if i == 0:
            np_X = img_row
            np_Y = result_Y_row        
        else:
            np_X = np.vstack((np_X,img_row))
            np_Y = np.vstack((np_Y,result_Y_row))
        f_tkr=open(tkr+'6b1_completed.txt','w+')
        f_tkr.close()
        
h5f_X.close()
h5f_Y.close()

#generate the message to the directory to signal the completion of this task
f=open('6b1_completed.txt','w+')
f.close()
