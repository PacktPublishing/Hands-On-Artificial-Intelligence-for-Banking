from pymongo import MongoClient
import json
import pprint

#client = MongoClient()
client = MongoClient('mongodb://localhost:27017/')
db_name = 'AIFinance8A'
collection_name = 'transactions_obp'

f_json = open('transactions.json', 'r')
json_data = json.loads(f_json)

db = client[db_name]
collection = db[collection_name]
posts = db.posts

result = posts.insert_many(json_data)

#to check if all documents are inserted
for post in posts.find({}):
    pprint.pprint(post)
