'''
step1: Copy the file to the import directory:
sudo cp dataset.csv /var/lib/neo4j/import/edge.csv
sudo cp product.csv /var/lib/neo4j/import/product.csv
sudo cp customer.csv /var/lib/neo4j/import/customer.csv

step2: at http://localhost:7474/browser/ from the browser
command 0: only do it first time
crete user name and password:
test, test

run the following commands:
command 1:
delete all nodes
MATCH (n) DETACH DELETE n;

command 2:
LOAD CSV WITH HEADERS FROM "file:///customer.csv" AS row
CREATE (c:Customer {customer_id: row.customer});

command 3:
LOAD CSV WITH HEADERS FROM "file:///product.csv" AS row
CREATE (p:Product {product_name: row.product});

command 4:
LOAD CSV WITH HEADERS FROM "file:///edge.csv" AS line
WITH line
MATCH (c:Customer {customer_id:line.customer})
MATCH (p:Product {product_name:line.product})
MERGE (c)-[:HAS {TYPE:line.type, VALUE:toInteger(line.value)}]->(p)
RETURN count(*);

command 5:
MATCH (c)-[cp]->(p) RETURN c,cp,p;

cheatsheet:
https://gist.github.com/DaniSancas/1d5265fc159a95ff457b940fc5046887
'''

#import libraries and define paramters
from neo4j import GraphDatabase
import spacy

#define the parameters, host, query and keywords
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("test", "test")) #enter your neo4j username and password instead of 'test'
session = driver.session()

check_q = ("MATCH (c:Customer)-[r:HAS]->(p:Product) " 
    "WHERE c.customer_id = $customerid AND p.product_name = $productname " 
    "RETURN DISTINCT properties(r)")

check_c = ("MATCH (c:Customer) " 
    "WHERE c.customer_id = $customerid " 
    "RETURN DISTINCT c")

update_q = ("MATCH (c:Customer)-[r:HAS]->(p:Product) " 
    "WHERE c.customer_id = $customerid AND p.product_name = $productname "
    "and r.TYPE = $attribute "
    "SET r.VALUE = toInteger($value) "
    "RETURN DISTINCT properties(r)")

intent_dict = {'check':check_q, 'login':check_c, 'update':update_q}

#list of key intent, product and attribute
product_list = ['deposit','loan']
attribute_list = ['pricing','balance']
intent_list = ['check','update']

print('loading nlp model')
nlp = spacy.load('en_core_web_md')
tokens_products = nlp(' '.join(product for product in product_list))
tokens_intent = nlp(' '.join(intent for intent in intent_list))
tokens_attribute = nlp(' '.join(attribute for attribute in attribute_list))

#define relevant functions to execute differnet queries
def run_query(tx, query, cid, product, attribute,attribute_val):
    result_list=[]
    for record in tx.run(query, customerid=cid,productname=product, attribute=attribute,value=attribute_val):
        result_list.append(record[0])
    return result_list


def intent_entity_attribute_extraction(nlp, sentence, tokens_intent, tokens_product, tokens_attribute):
    #please implement your sentence classification here to extract the intent
    tokens = nlp(sentence)
    #use the NER to extract the entity regarding product

    intent_score= 0
    product_score = 0
    attribute_score = 0
    final_intent = ''
    final_product = ''
    final_attribute = ''
    final_attribute_value = ''

    threshold = 0.8

    for token in tokens:
        for intent in tokens_intent:
            curr_intent_score = token.similarity(intent)
            if curr_intent_score > intent_score and curr_intent_score > threshold:
                intent_score = curr_intent_score
                final_intent = intent.text

        for product in tokens_product:
            curr_product_score = token.similarity(product)
            if curr_product_score > product_score and curr_product_score > threshold:
                product_score = curr_product_score
                final_product = product.text
                    
        for attribute in tokens_attribute:
            curr_attribute_score = token.similarity(attribute)
            if curr_attribute_score > attribute_score and curr_attribute_score > threshold:
                attribute_score = curr_attribute_score
                final_attribute = attribute.text

        if token.pos_ == 'NUM' and token.text.isdigit():
            final_attribute_value = token.text

    print('matching...')
    print(final_intent, final_product, final_attribute, final_attribute_value)
    return (final_intent, final_product, final_attribute, final_attribute_value)

name = ''
product = ''
attribute = ''
attribute_val = ''
reset = False

while True:
    
    #Authentication
    if name == '' or reset:
        name = input('Hello, What is your name? ')
        print('Hi '+name)
        #check for login
        query_str = intent_dict['login']
        result = session.read_transaction(run_query, query_str, name, product,attribute,attribute_val)
        if len(result)==0:
            print('Failed to find '+name)
            print('Press http://packtpub.com to register your account')
            name =''
            continue
    
    #Sentences Intent and Entities Extraction
    input_sentence = input('What do you like to do? ')
    if input_sentence == "reset":
        reset = True 
    entities = intent_entity_attribute_extraction(nlp, input_sentence,tokens_intent, tokens_products, tokens_attribute)
    #actually can build another intent classifier here based on the scores and words matched as features, as well as previous entities
    intent = entities[0]
    product = entities[1]
    attribute = entities[2]
    attribute_val = entities[3]

    #cross-check for missing information
    while intent == '':
        input_sentence = input('What do you want to do?')
        entities = intent_entity_attribute_extraction(nlp, input_sentence,tokens_intent, tokens_products, tokens_attribute)
        intent = entities[0]

    while product == '':
        input_sentence = input('What product do you want to check?')
        entities = intent_entity_attribute_extraction(nlp, input_sentence,tokens_intent, tokens_products, tokens_attribute)
        product = entities[1]

    while attribute == '':
        input_sentence = input('What attribute of the ' + product +' that you want to '+intent+'?')
        entities = intent_entity_attribute_extraction(nlp, input_sentence,tokens_intent, tokens_products, tokens_attribute)
        attribute = entities[2]

    #execute the query to extract the answer
    query_str = intent_dict[intent]
    results = session.read_transaction(run_query, query_str, name,product,attribute,attribute_val)
    if len(results) >0:
        for result in results:
            if result['TYPE'] == attribute:
                print(attribute + ' of ' + product + ' is '+str(result['VALUE']))
    else:
        print('no record')
