import json
import os
import re
import sqlite3
from bs4 import BeautifulSoup
import requests

import lib_cnt_sentiment as sentiment

#load tweet json
script_dir = os.path.dirname(__file__)
rel_path = 'search/data.json'
json_abs_file_path = os.path.join(script_dir,rel_path)

with open(json_abs_file_path) as f:
    data = json.load(f)

#db file
db_path = 'parsed_search.db'
db_name = 'cse_db'

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
sqlstr = "CREATE TABLE IF NOT EXISTS "+db_name+"(org_txt TEXT, txt_dte DATE,source_type TEXT, source_impact REAL,topics TEXT,positive REAL, negative REAL, link_positive REAL, link_negative REAL)"
c.execute(sqlstr)
conn.commit()

#select_fld = ['title','link', results[0]['pagemap']['metatags'][0]['parsely-pub-date'], 'snippet']

#loop through the json and insert the result to the db
print_cnt = 0

'''
print('sentiment analysis and topic modeling')
pos,neg = sentiment.cnt_sentiment(text)

if len(text) > 100000:
    text = text[:100000-1]

print('ner')
common_words, sentences,keywords = sentiment.NER(text)

list_sentences = sentiment.prep_sentence_list(keywords)
'''

#format at https://developers.google.com/custom-search/v1/cse/list#response
for search in data:
    search_link_txt_pos=0
    search_link_txt_neg=0
    search_txt = search['snippet']
    search_txt_pos,search_txt_neg = sentiment.cnt_sentiment(search_txt)
    keywords,sentences_list,words_list = sentiment.NER_topics(search_txt)
    url_link = search['link']
    if len(url_link)>0:
        url_txt = sentiment.url_to_string(url_link)
        temp_search_link_txt_pos, temp_search_link_txt_neg = sentiment.cnt_sentiment(url_txt)
        search_link_txt_pos+=temp_search_link_txt_pos
        search_link_txt_neg+=temp_search_link_txt_neg
    search_usr_description = search['pagemap']['metatags'][0]["twitter:site"]
    search_usr_followers = 0
    #bloomberg
    try:
        search_time = search['pagemap']['metatags'][0]["validatedat"]
    except Exception:
        #reuters
        search_time = search['pagemap']['metatags'][0]["og:article:modified_time"]
    
    item = [(search_txt,search_time,search_usr_description,search_usr_followers,str(words_list),search_txt_pos,search_txt_neg,search_link_txt_pos,search_link_txt_neg)]
    sqlstr = 'insert into '+db_name+'(org_txt, txt_dte,source_type, source_impact,topics,positive,negative,link_positive,link_negative) VALUES (?,?,?,?,?,?,?,?,?)'
    c.executemany(sqlstr, item)
    print_cnt+=1
    if print_cnt % 10 == 0 or len(data) <10:
        conn.commit()
        print(str(print_cnt))

    print(print_cnt)

#closing out and output the results
c.close()
