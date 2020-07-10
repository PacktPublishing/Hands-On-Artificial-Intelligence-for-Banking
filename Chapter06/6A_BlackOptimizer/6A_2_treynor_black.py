#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 22:46:14 2018

@author: jeff
"""
'''*************************************
#1. Import libraries and key variable values

'''
import sqlite3
import datetime

now_dt= datetime.datetime.now()
now_str = now_dt.strftime('%Y%m%d%H%M%S')

conn = sqlite3.connect('ETF_Para.db')
c = conn.cursor()
#create a table to store weight
drop = True

if drop== True:
    print('drop table')
    sqlstr = "drop table ETF_para_weight"
    try:
        output = c.execute(sqlstr)
    except Exception:
        print('non exists')

print('create')
sqlstr = "CREATE TABLE IF NOT EXISTS ETF_para_weight(  TICKER TEXT PRIMARY KEY,  weight REAL, timestamp)"
c.execute(sqlstr)
conn.commit()

'''*************************************
#2. Find out the weight of the securities in the active portfolio
'''
#total alpha/variance of the active securities
print('para')
sqlstr = "select sum(alpha / var_err) from ETF_para"
c.execute(sqlstr)
result = c.fetchall()
total=result[0][0]

print(total)
conn.commit()

#insert into the table the weight of each active securities
print('insert')
sqlstr = "insert into ETF_para_weight select TICKER, (alpha / var_err)/"+str(total)+","+now_str + " from ETF_para"
c.execute(sqlstr)
conn.commit()

'''*************************************
#3. Find out the weight of the active portfolio in the total portfolio
'''
print('cal')
#calculate the parameters of the active portfolio
sqlstr = "select sum(a.alpha * b.weight) as alpha_A,sum(a.beta * b.weight) as beta_A,sum(a.var_err * b.weight) as var_err_A from ETF_para as a left join ETF_para_weight as b on a.TICKER = b.TICKER"
c.execute(sqlstr)
portfolio_A = c.fetchall()[0]
conn.commit()
alpha_A = portfolio_A[0]
beta_A = portfolio_A[1]
var_err_A = portfolio_A[2]
#read back the risk free and market para
f = open('r.txt','r')
r = f.read()
rates = r.split(',')
r_f=float(rates[0])
r_m=float(rates[1])
var_m=float(rates[1])
#calculate the weight of active portfolio
w_0 = (alpha_A/var_err_A)/ ((r_m-r_f)/var_m)
w_A = w_0/ (1+(1-beta_A)*w_0)

#display the result
print('We should allocate '+str(w_A) +' to active portfolio')
c.close()

'''
0.5010161431130361

at OS - kill the sqlite connection at terminal in case the job was interrupted and database locked
fuser ETF_Para.db #find out the process id XXX
kill -9 XXXX
'''
