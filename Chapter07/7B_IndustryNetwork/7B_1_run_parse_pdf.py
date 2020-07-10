'''*************************************
#1. Import relevant libraries and variables

'''
#custom made function
import lib_entitiesExtraction as entitiesExtraction
import lib_parser_pdf as pdf_parser
import json
import sqlite3

pdf_path = 'annualrpt/NYSE_DUK_2017.pdf'

print('parse PDF')
text_org = pdf_parser.convert_pdf_to_txt(pdf_path)

print('text cleansing')
text = text_org.replace('\r', ' ').replace('\n', ' ').replace('\s',' ')

'''*************************************
#2. NLP

'''
#Named Entity Extraction
print('ner')
#see if we need to convert everthing to lower case words - we keep the original format for this case
lower=False
common_words, sentences, words_list,verbs_list = entitiesExtraction.NER_topics(text,lower)
entities_in_sentences = entitiesExtraction.org_extraction(text)

ents_dict = {}

sentence_cnt = 0

#create this list to export the list of ent and cleanse them
f_ent = open('entities.txt','w+')
ent_list=[]

print('looping sentences')
for sentence in entities_in_sentences:
    ents_dict[sentence_cnt] = {}
    for entity in sentence:
        ent_type = entity[1]
        ent_name = entity[0]
        ent_name = ent_name.strip()

        if len(ent_name)==0 or ent_name =='':
            continue
        
        if lower == True:
            ent_name = ent_name.lower()
            
        if ent_type in( 'ORG','PERSON','FAC','NORP','GPE','LOC','PRODUCT'):
            #take only upper case (1st pos)
            if ent_name[0].islower():
                continue

            if ent_type not in ents_dict[sentence_cnt]:
                ents_dict[sentence_cnt][ent_type]=[]

            ents_dict[sentence_cnt][ent_type].append(ent_name)

            if ent_name not in ent_list:
                ent_list.append(ent_name)
                f_ent.write(ent_name+'\t'+ent_name+'\n')
            ents_dict[sentence_cnt]['VERB'] = verbs_list[sentence_cnt]
        else:
            ents_dict[sentence_cnt][ent_type] = []

    #handle other type
    for entity in sentence:
        ent_type = entity[1]
        ent_name = entity[0]
        ent_name = ent_name.strip()
        if len(ent_name)==0 or ent_name =='':
            continue
        if ent_type not in('ORG','PERSON','FAC','NORP','GPE','LOC','PRODUCT'):
            ents_dict[sentence_cnt][ent_type].append(ent_name)

    sentence_cnt+=1

f_ent.close()

#Insert the entities into SQL database
print('insert')

f = open('result.json','w+')
json.dump(ents_dict,f)

#db file
db_path = 'parsed_network.db'
db_name = 'network_db'
db_name2 = 'sentence_db'

#sql db
conn = sqlite3.connect(db_path)
c = conn.cursor()

sqlstr = "drop table "+db_name
try:
    output = c.execute(sqlstr)
except Exception:
    print('non exists')

sqlstr = "drop table "+db_name2
try:
    output = c.execute(sqlstr)
except Exception:
    print('non exists')


entity_types = ['ORG','PERSON','FAC','NORP','GPE','LOC','PRODUCT']
relation_units = ['DATE','TIME','PERCENT','MONEY','QUANTITY','ORDINAL','CARDINAL']
#create if required
print('create')
sqlstr = "CREATE TABLE IF NOT EXISTS "+db_name+"(SOURCE TEXT, SENTENCE_NO INTEGER, CON_TYPE TEXT, ENTITY TEXT, RELATION_UNIT TEXT, RELATION_VAL TEXT,VERB_LIST TEXT )"
c.execute(sqlstr)
conn.commit()
sqlstr = "CREATE TABLE IF NOT EXISTS "+db_name2+"(SOURCE TEXT, SENTENCE_NO INTEGER, SENTENCE TEXT )"
c.execute(sqlstr)
conn.commit()

for key, value in ents_dict.items():
    con_type=''
    entity=''
    relation_unit = ''
    relation_val = ''
    verb_list = ''
    #key = sentence number
    sent_item = [(pdf_path, key, str(value))]
    sqlstr = 'insert into '+db_name2+'(SOURCE, SENTENCE_NO, SENTENCE) VALUES (?,?,?)'
    c.executemany(sqlstr, sent_item)
    
    for sub_key,sub_value in ents_dict[key].items():
        for entity_type in entity_types:
            try:
                entity = str(ents_dict[key][entity_type])[1:-1]
                con_type = entity_type
                verb_list = str(ents_dict[key]['VERB'])[1:-1]
            except Exception:
                continue
        for relation in relation_units:
            try:
                relation_val = ents_dict[key][relation][0]
                relation_unit = relation
            except Exception:
                continue
    if len(con_type+entity+relation_unit+relation_val+verb_list)>0:
        item = [(pdf_path, key, con_type,entity,relation_unit,relation_val,verb_list)]
        sqlstr = 'insert into '+db_name+'(SOURCE, SENTENCE_NO, CON_TYPE, ENTITY, RELATION_UNIT,RELATION_VAL,VERB_LIST) VALUES (?,?,?,?,?,?,?)'
        c.executemany(sqlstr, item)
    if key % 10 == 0:
        conn.commit()
    
#closing out and output the results
conn.commit()
c.close()
