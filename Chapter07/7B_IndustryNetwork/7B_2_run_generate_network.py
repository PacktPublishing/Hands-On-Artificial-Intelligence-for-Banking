'''*************************************
#1. Import relevant libraries and variables

'''
#generate network
import sqlite3
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

#db file
db_path = 'parsed_network.db'
db_name = 'network_db'

#sql db
conn = sqlite3.connect(db_path)
c = conn.cursor()

sql_str = "select source, entity,CON_TYPE from network_db where con_type in ('ORG','PERSON','FAC','NORP','GPE','LOC','PRODUCT') and verb_list <> '[]' group by source,CON_TYPE, entity"
df_org = pd.read_sql_query(sql_str, conn)

network_dict={}
edge_list=[]
curr_source =''
curr_entity = ''
org_list = []
person_list = []

'''*************************************
#2. generate the network with all entities connected to Duke Energy - whose annual report is parsed

'''
target_name = 'Duke Energy'
#loop through the database to generate the network format data
for index, row in df_org.iterrows():
    if curr_entity != row['SOURCE']:
        curr_source = row['SOURCE']
        network_dict[curr_source]=[]
    curr_entity = row['ENTITY']
    if len(curr_entity)==0:
        continue
    curr_entities = curr_entity.split(',')
    if len(curr_entities)>1:
        entity1=curr_entities[0].strip()
        if entity1 == '' or entity1=="'":
            curr_entities=curr_entities[1:]
            entity1 = curr_entities[0]
        network_dict[entity1]=[]
        org_list.append(entity1)
        #all entity has relationship with target company
        entry = (target_name,entity1,{'weight':1})
        edge_list.append(entry)
        for entity in curr_entities[1:]:
            entity2=entity.strip()
            if entity2=='' or entity2 =="'":
                continue
            network_dict[entity1].append(entity2)
            entry = (entity1,entity2,{'weight':1})
            edge_list.append(entry)
            org_list.append(entity2)

            entry = (target_name,entity2,{'weight':1})
            edge_list.append(entry)

#Generate the output in networkX
print('networkx')

#output the network
G = nx.from_edgelist(edge_list)
pos = nx.spring_layout(G)
nx.draw(G, with_labels=False, nodecolor='r',pos=pos, edge_color='b')
plt.savefig('network.png')

#Generate output for Neo4J
print('prep data for neo4j')
f_org_node=open('node.csv','w+')
f_org_node.write('nodename\n')

f_person_node=open('node_person.csv','w+')
f_person_node.write('nodename\n')

f_vertex=open('edge.csv','w+')
f_vertex.write('nodename1,nodename2,weight\n')

unique_org = set(org_list)
for entity in unique_org:
    f_org_node.write(entity+'\n')
f_org_node.close()

unique_person = set(person_list)
for entity in unique_person:
    f_person_node.write(entity+'\n')
f_person_node.close()

for edge in edge_list:
    node1, node2, weight_dict = edge
    weight = weight_dict['weight']
    f_vertex.write(node1+','+node2+','+str(weight)+'\n')
f_vertex.close()

'''
What to do afterwards:
step1: Copy the file to the import directory:

sudo cp '/home/jeff/AI_Finance_book/7B_IndustryNetwork/edge.csv' /var/lib/neo4j/import/edge.csv
sudo cp '/home/jeff/AI_Finance_book/7B_IndustryNetwork/node.csv' /var/lib/neo4j/import/node.csv

step2: start the neo4j server
sudo service neo4j restart

check if the server is runnning
journalctl -e -u neo4j

step3: use the browser to go to the neo4j server
http://localhost:7474/browser/
https://neo4j.com/developer/guide-importing-data-and-etl/#_exporting_the_data_to_csv

step4: run the following scripts at neo4j browser
MATCH (n) DETACH DELETE n;

USING PERIODIC COMMIT
LOAD CSV WITH HEADERS FROM "file:///node.csv" AS row
CREATE (:ENTITY {node: row.nodename});

CREATE INDEX ON :ENTITY(node);


USING PERIODIC COMMIT
LOAD CSV WITH HEADERS FROM "file:///edge.csv" AS row
MATCH (vertex1:ENTITY {node: row.nodename1})
MATCH (vertex2:ENTITY {node: row.nodename2})
MERGE (vertex1)-[:LINK]->(vertex2);

MATCH (n:ENTITY)-[:LINK]->(ENTITY) RETURN n;

'''

