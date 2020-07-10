#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Wed Jan  2 00:12:51 2019

@author: jeff
'''
#define libraries and variables
import sqlite3
from pymongo import MongoClient
import json
from flatten_dict import flatten

client = MongoClient('mongodb://localhost:27017/')
db_name = 'AIFinance8A'
collection_name = 'transactions_obp'

db = client[db_name]
collection = db[collection_name]
posts = db.posts

posts_dict = posts.find({})

input_dict = posts_dict

#flatten the dictionary
entries_dict = {}
entry_cnt = 0

for one_dict in input_dict:
    entries_dict[entry_cnt] = flatten(one_dict)
    entry_cnt+=1

tuple_fields_list= entries_dict[0].keys()
str_fields_list=[]

for fld in tuple_fields_list:
    fld_str = '_'.join(fld)
    str_fields_list.append(fld_str)


#create the database schema
#db file
db_path = 'parsed_obp.db'
db_name = 'obp_db'

#sql db
conn = sqlite3.connect(db_path)
c = conn.cursor()

sqlstr = 'drop table '+db_name
try:
    output = c.execute(sqlstr)
except Exception:
    print('non exists')

print('create')
fields_str = ('(_id TEXT ,details_completed TEXT ,details_description TEXT ,details_new_balance_amount TEXT ,'
    'details_new_balance_currency TEXT ,details_posted TEXT ,details_type TEXT ,details_value_amount TEXT ,details_value_currency TEXT ,id TEXT, '
    'metadata_comments TEXT ,metadata_images TEXT ,metadata_narrative TEXT ,metadata_tags TEXT ,metadata_where TEXT ,other_account_bank_name TEXT ,'
    'other_account_bank_national_identifier TEXT ,other_account_holder_is_alias TEXT ,other_account_holder_name TEXT ,other_account_IBAN TEXT ,'
    'other_account_id TEXT ,other_account_kind TEXT ,other_account_metadata_corporate_location TEXT ,other_account_metadata_image_URL TEXT ,'
    'other_account_metadata_more_info TEXT ,other_account_metadata_open_corporates_URL TEXT ,other_account_metadata_physical_location TEXT ,'
    'other_account_metadata_private_alias TEXT ,other_account_metadata_public_alias TEXT ,other_account_metadata_URL TEXT ,other_account_number TEXT '
    ',other_account_swift_bic TEXT ,this_account_bank_name TEXT ,this_account_bank_national_identifier TEXT ,this_account_holders TEXT ,this_account_IBAN TEXT ,'
    'this_account_id TEXT ,this_account_kind TEXT ,this_account_number TEXT ,this_account_swift_bic TEXT)')
sqlstr = 'CREATE TABLE IF NOT EXISTS '+db_name+ fields_str
    
c.execute(sqlstr)
conn.commit()
print_cnt = 0

#loop through the dict and insert them into the db
dict_cnt = entries_dict.keys()

for cnt in dict_cnt:
    fld_list_str = ''
    fld_list_tuple = ()
    length_fld =0
    for fld in tuple_fields_list:
        fld_str = '_'.join(fld)
        entry_val = entries_dict[cnt][fld]
        if entry_val is None:
            entry_val = ' '
        if type(entry_val) == "<class 'list'>":
            entry_val = str(entry_val)
        fld_list_tuple+=(str(entry_val),)
        length_fld+=1
        fld_list_str += ',' + fld_str
    fld_list_str = fld_list_str[1:]
    item = [fld_list_tuple]
    length_fld = len(tuple_fields_list)
    question_len = ',?'*length_fld
    sqlstr = 'insert into '+ db_name+ '(' + str(fld_list_str)+') VALUES ('+question_len[1:]+')'
    #print(sqlstr)
    #print(item)
    c.executemany(sqlstr, item)
    print_cnt+=1
    if print_cnt % 10 == 0:
        conn.commit()
        print(str(print_cnt))
