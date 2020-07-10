#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 07:38:32 2018

@author: jeff
"""
'''*************************************
#1. Import libraries and key varable values

'''

import json
import os
import re
import sqlite3
import lib_cnt_sentiment as sentiment



#db file
db_path = 'parsed_tweets.db'
db_name = 'tweet_db'

#sql db
conn = sqlite3.connect(db_path)
c = conn.cursor()

sqlstr = "drop table "+db_name
try:
    output = c.execute(sqlstr)
except Exception:
    print('non exists')

#create if required
print('create')
sqlstr = "CREATE TABLE IF NOT EXISTS "+db_name+"(securities TEXT, org_txt TEXT, txt_dte DATE,source_type TEXT, source_impact REAL,topics TEXT, positive REAL, negative REAL, link_positive REAL, link_negative REAL)"
c.execute(sqlstr)
conn.commit()


#load tweet json
script_dir = os.path.dirname(__file__)
rel_path = 'twitter'
file_list = os.listdir(os.path.join(script_dir,rel_path))

#loop through the tweets
for filename in file_list:
    if not filename.endswith('.json'):
        continue
        
    print(filename)
    json_abs_file_path = os.path.join(script_dir,rel_path,filename)

    f = open(json_abs_file_path)
    data = json.load(f)
    
    #loop through the json and insert the result to the db
    print_cnt = 0
    link_words_list = []

    for tweet in data:
        tweet_link_txt_pos=0
        tweet_link_txt_neg=0
        tweet_txt = tweet['text']
        tweet_txt_pos,tweet_txt_neg = sentiment.cnt_sentiment(tweet_txt)
        keywords,sentences_list,words_list = sentiment.NER_topics(tweet_txt)
        url_link = tweet['entities']['urls']
        if len(url_link)>0:
            url = [url_link[0]['url']]
            url_txt = sentiment.url_to_string(url)
            temp_tweet_link_txt_pos, temp_tweet_link_txt_neg = sentiment.cnt_sentiment(url_txt)
            link_keywords,link_sentences_list,link_words_list = sentiment.NER_topics(tweet_txt)
            tweet_link_txt_pos+=temp_tweet_link_txt_pos
            tweet_link_txt_neg+=temp_tweet_link_txt_neg
        tweet_usr_description = tweet['user']['description']
        tweet_usr_followers = tweet['user']['followers_count']
        tweet_time = tweet['created_at']
        words_list = words_list + link_words_list
        item = [(filename, tweet_txt,tweet_time,tweet_usr_description,tweet_usr_followers,str(words_list),tweet_txt_pos,tweet_txt_neg,tweet_link_txt_pos,tweet_link_txt_neg)]
        sqlstr = 'insert into tweet_db(securities, org_txt, txt_dte,source_type, source_impact,topics, positive,negative,link_positive,link_negative) VALUES (?,?,?,?,?,?,?,?,?,?)'
        c.executemany(sqlstr, item)
        print_cnt+=1
        if print_cnt % 10 == 0:
            conn.commit()
            print(str(print_cnt))

#closing out and output the results
conn.commit()
c.close()
