'''*************************************
#1. Import libraries and key varable values

'''
import sqlite3
import pandas as pd
import plotly
import plotly.graph_objs as go
import quandl
import json


# Create your connection.
db_path = 'parsed_tweets.db'
cnx = sqlite3.connect(db_path)
db_name = 'tweet_db'

'''
Mon Nov 26 22:57:10 +0000 2018
123456789112345678901234567890
'''
'''*************************************
#2. Gauge the sentiment of each security

'''
sql_str = 'SELECT securities FROM '+ db_name +' group by securities'
df_securities = pd.read_sql_query(sql_str, cnx)
sec_list = df_securities['securities'].unique()
data=[]

sql_str = 'select yr_mth_day, b.* from (SELECT substr(txt_dte,27,4) || substr(txt_dte,5,3)||substr(txt_dte,9,2) as yr_mth_day from ' + db_name+' group by substr(txt_dte,27,4) || substr(txt_dte,5,3)||substr(txt_dte,9,2)) as a'+ ' left join (Select securities, substr(txt_dte,27,4) || substr(txt_dte,5,3)||substr(txt_dte,9,2) as b_yr_mth_day, positive as positive,negative as negative, link_positive as l_positive, link_negative as l_negative FROM '+ db_name+ ') as b'+ ' on a.yr_mth_day = b.b_yr_mth_day'
df = pd.read_sql_query(sql_str, cnx)
df['total_sentiment'] = (df['positive'] - df['negative'] + df['l_positive'] - df['l_negative'])/(df['positive'] + df['negative'] + df['l_positive'] + df['l_negative'])
df_gp = df.groupby(['securities','yr_mth_day'],as_index =False)['total_sentiment'].mean()

print('Sentiment across securities')
field_list = ['positive','negative']
for sec in sec_list:
    df_select = df_gp[df_gp['securities']==sec]
    data_select = go.Scatter(
            x=df_select['yr_mth_day'], # assign x as the dataframe column 'x'
            y=df_select['total_sentiment'],
            mode = 'lines+markers',
            name = sec
            )
    data.append(data_select)
print('ploting')
plotly.offline.plot(data, filename='sample_output.html')


'''*************************************
#3. Compare sentiment against price

'''
print('Price vs sentiment')
quandl.ApiConfig.api_key = '[quandl id]'
date_range={'gte':2018-1-1}

#define the function to do so
def price_sentiment(tkr, target_sec,date_range):
    print('"'+tkr+'"')
    print('"'+target_sec+'"')
    record_db = quandl.get_table('SHARADAR/SEP', date=date_range, ticker=tkr)
    record_db['mid'] = (record_db['high']+record_db['low'])/2
    record_db = record_db.sort_values(by=['date'])
    print(record_db.shape)
    if record_db.shape[0]==0:
        print('no data1')
        return
    
    #show the header
    print(record_db.head(0))

    #format specified in http://strftime.org/
    record_db['yr_mth_day']=record_db['date'].apply(lambda x: x.strftime('%Y%b%d')) 

    df_tweet = df_gp[df_gp['securities']==target_sec]
    if df_tweet.shape[0]==0:
        print('no data2')
        return
    
    #plot the data
    df_plot = pd.merge(record_db,df_tweet, how='inner', on='yr_mth_day')
    
    
    price_sentiment_data = []
    data_select = go.Scatter(
                x=df_plot['yr_mth_day'], # assign x as the dataframe column 'x'
                y=df_plot['mid'],
                mode = 'lines+markers',
                name = 'price'
                )
    price_sentiment_data.append(data_select)
    data_select = go.Scatter(
                x=df_plot['yr_mth_day'], # assign x as the dataframe column 'x'
                y=df_plot['total_sentiment'],
                mode = 'lines+markers',
                name = 'sentiment',
                yaxis='y2'
                )
    price_sentiment_data.append(data_select)
    print('ploting')
    #Plot sentiment vs stock price in different axis
    layout = go.Layout(
        title=target_sec + ' Price vs Sentiment',
        yaxis=dict(
            title='yaxis title'
        ),
        yaxis2=dict(
            title='yaxis2 title',
            titlefont=dict(
                color='rgb(148, 103, 189)'
            ),
            tickfont=dict(
                color='rgb(148, 103, 189)'
            ),
            overlaying='y',
            side='right'
        )
    )
    fig = go.Figure(data=price_sentiment_data, layout=layout)
    plotly.offline.plot(fig, filename=tkr+'priceVSsentiment2.html')

    #correlation
    corr_res = df_plot[['total_sentiment','close']].corr()
    print(corr_res)
    return corr_res

#run it on different companies
print('Retrieve data')
df_comp = pd.read_csv('ticker_companyname.csv')
corr_results={}

for index, row in df_comp.iterrows():
    tkr = row['ticker']
    name = row['name']

    target_sec = '"'+name +'"data.json'
    
    corr_result = price_sentiment(tkr,target_sec,date_range)
    try:
        corr_results[name]=corr_result['close'][0]
    except Exception:
        continue

f_corr = open('corr_results.json','w')
json.dump(corr_results,f_corr)
f_corr.close()

