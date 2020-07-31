#!/usr/bin/env python3
# -*- coding: utf-8 -*-
QUANDLKEY = '<ENTER YOUR QUANDLKEY HERE>'
"""
Created on Wed Oct  3 01:43:48 2018

@author: jeff
"""
#Step 2 and 3. Perform clusterings to find out the similar investors whose sharing the similar stocks
#import relevant libraries
import os
import pickle
import math
import numpy as np
import pandas as pd
import quandl

#input this file name as an output from 5A_S1_1b1c1d_investor_similarity.py
INVESTOR_CLUSTER = 'investor_labels_482.pkl'
list_fld = ['ps1','pe1','pb'
            ,'marketcap'
            ,'divyield','bvps'
            ,'ebitda','grossmargin','netmargin','roa','roe'
            ,'de']

#perform financial projection
#reuse the function developed for WACC optimization
def cal_F_financials(record_db_f, logreg, logreg_sc, new_debt_pct,price_offering,levered_beta,sales_growth,coefs,r_free):
    new_equity_pct = 1- new_debt_pct

    coef_sales_ppenet = coefs[0]
    coef_sales_cor	= coefs[1]
    coef_cor_opex = coefs[2]

    
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

    F_ps1	=	price_offering	/(F_sales/(unit_offering+record_db_f['shareswa'][0]))
    F_pe1	=	price_offering	/(F_earnings	/	 (unit_offering+record_db_f['shareswa'][0]))
    F_pb	=	price_offering	/ (F_equity	/	 (unit_offering+record_db_f['shareswa'][0]))
    F_marketcap	=	price_offering	*	 (unit_offering+record_db_f['shareswa'][0])	
    F_divyield	=	record_db_f['divyield'][0]			
    F_bvps	=	F_equity	/	 (unit_offering+record_db_f['shareswa'][0])	
    F_ebitda	=	F_ebitda			
    F_grossmargin	=	F_gross_profit			
    F_netmargin	=	F_earnings	/	F_sales	
    F_roa	=	F_earnings	/	F_asset	
    F_roe	=	F_earnings	/	F_equity	
    F_de	=	F_D_E

    metric_list = [F_ps1,F_pe1,F_pb,F_marketcap,F_divyield,F_bvps,F_ebitda,F_grossmargin,F_netmargin,F_roa,F_roe,F_de]

    return metric_list

'''
flds=['ps1','pe1','pb'
        ,'marketcap'
        ,'divyield','bvps'
        ,'ebitda','grossmargin','netmargin','roa','roe'
        ,'de']
'''

'''*****************************
Step 2: Simulate financial of the the new stock
'''
current_file_dir = os.path.dirname(__file__)
file_path = os.path.join(current_file_dir,'data','stock_data')

#load credit model built previoiusly
files = os.listdir(file_path)

f_logreg=open('log_reg.pkl',"rb")
logreg = pickle.load( f_logreg)

f_logreg_sc=open('logreg_scaler.pkl',"rb")
logreg_sc = pickle.load( f_logreg_sc)    

#reuse the parameters developed from WACC example
levered_beta = 0.24582618
sales_growth = 0.10

coef_sales_ppenet = -0.00012991
coef_sales_cor	= 0.69804978
coef_cor_opex = 0.35883998
coefs = [coef_sales_ppenet,coef_sales_cor,coef_cor_opex]

r_free = 0.00244132    

#assume that we are raising equity for the same client
tkr = 'DUK'
#sicindustry = "ELECTRIC & OTHER SERVICES COMBINED"
sicindustry = 'TELEPHONE COMMUNICATIONS (NO RADIOTELEPHONE)'
quandl.ApiConfig.api_key = QUANDLKEY
record_db_f=quandl.get_table('SHARADAR/SF1', calendardate='2017-12-31', ticker=tkr,dimension='MRY')

new_debt_pct=0.86
price_offering=84.11

#run simulation / projection of financial data
list_metrics = cal_F_financials(record_db_f, logreg, logreg_sc, new_debt_pct,price_offering,levered_beta,sales_growth,coefs,r_free)
np_metrics = np.array(list_metrics).reshape(1,-1)

'''*****************************
Step 3: Run the similarity models to find out holders of the similar stocks
'''
#check if we need any model - if industy has too few stocks, no model needed to find out the similar stocks
has_cluster_model = False
for file in files:
    stock_names = file.split('_')
    if stock_names[0] == sicindustry:
        if stock_names[1].startswith('cluster'):
            cluster_file_path = os.path.join(file_path, file)
            f_cluster = open(cluster_file_path,'rb')
            the_cluster = pickle.load(f_cluster)
            has_cluster_model = True
        elif stock_names[1].startswith('SC'):
            sc_file_path = os.path.join(file_path, file)
            f_scaler = open(sc_file_path,'rb')
            the_scaler = pickle.load(f_scaler)
        else:
            labels_file_path = os.path.join(file_path, file)
            f_labels = open(labels_file_path,'rb')
            the_labels = pickle.load(f_labels)

#retrieve the list of tickers that are similar
if has_cluster_model:
    X = the_scaler.transform(np_metrics)
    X_df = pd.DataFrame(X)
    label = the_cluster.predict(X_df)
    the_labels_select = the_labels[the_labels['cluster'] == label[0]]
    list_stock_scope = list(the_labels_select['ticker'])
else:
    #go to the industry direcctly
    groupby_fld = 'sicindustry'
    df_tkr = pd.read_csv('industry_tickers_list.csv')
    
    #filter the industry in scope
    df_tkr_ind_select = df_tkr[df_tkr[groupby_fld]==sicindustry]
    list_stock_scope = list(df_tkr_ind_select['ticker'])
    #no cluster model

#list_scope = ['AMOV']
list_investors =[]
dte_str = '2018-03-31'

#find list of investors looking at the similar size and more
#check which investors have it...
stock_file_path = os.path.join(current_file_dir,'data','investor_data')
investorNameList = os.listdir(stock_file_path)

#loop through investors holding name by name to find out investor that is holding the similar stocks
for filename in investorNameList:
    tmp_name = filename.split('.')
    if tmp_name[1] == 'csv':
        investor = tmp_name[0]
        data_df = quandl.get_table("SHARADAR/SF3", paginate=True,investorname=investor,calendardate=dte_str)
        for stock in list_stock_scope:
            stock_df = data_df[data_df['ticker'] == stock]
            if len(stock_df) > 0:
                list_investors.append(investor)

#Load the investor clustering model
pd_investorname = pd.read_csv(filepath_or_buffer='investor_summary.csv',header=3)

investor_cluster_file = INVESTOR_CLUSTER
f_investor_cluster=open(investor_cluster_file,"rb")
investor_label_df = pickle.load( f_investor_cluster)

investor_fld = 'investorname_x'

#extract the investors's cluster id
investor_df_select = pd_investorname[investor_fld].isin(list_investors)
investor_df_select=investor_df_select.rename('isInvestor')
result_combined = pd.concat([pd_investorname,investor_df_select,investor_label_df['cluster']], axis=1, sort=False)
cluster_select = result_combined.loc[(result_combined['isInvestor'] == True)]

#find out who else share the same cluster id
list_cluster = cluster_select['cluster'].unique()
select_investor_df = result_combined['cluster'].isin(list_cluster)
final_investor_list = list(result_combined[investor_fld])

#print out the investor list
print(str(final_investor_list))
