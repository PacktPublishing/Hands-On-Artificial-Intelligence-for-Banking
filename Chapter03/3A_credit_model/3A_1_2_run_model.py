'''*************************************
1a. import all the libraries required
'''
import os
import re
import pandas as pd
import pickle
'''*************************************
2. Define program-wide variables and values
'''
#set up the working directory where datafiles are all located
data_path = os.getcwd()
os.chdir(data_path)

#read in files
file_attrib_in = os.path.join(data_path,'attrib.txt')
file_path_in = os.path.join(data_path,'5year.csv')
f_name = pd.read_csv(file_path_in,sep=',')
f_attrib = open(file_attrib_in,"r")
attrib_str = f_attrib.read()

#assign column headers
label_name = 'default'
re_obj = re.compile(r'X[0-9]+\s')
fields_list = re_obj.split(attrib_str)
fields_list = fields_list[1:]
fields_list.append(label_name)
f_name.columns = fields_list

#create X and Y dataset with the right header
X = f_name.iloc[:,:-1]
Y= f_name.iloc[:,-1]
#make sure data types are correct and missing values are handled
Y=Y.astype(int)
cols = X.columns[X.dtypes.eq(object)]
for c in cols:
    X[c] = pd.to_numeric(X[c], errors='coerce')
X=X.fillna(0)

f_logreg=open('log_reg.pkl',"rb")
logreg = pickle.load( f_logreg)

f_logreg_sc=open('logreg_scaler.pkl',"rb")
logreg_sc = pickle.load( f_logreg_sc)

best_col_list=[]
thefile = open('logreg_cols.txt', 'r')
for line in thefile:
    best_col_list.append(line.rstrip('\n'))

def select_columns(df, col_list):
    df_selected = df[df.columns.intersection(col_list)]
    return df_selected

X_selected = select_columns(X,best_col_list)
X_selected = logreg_sc.transform(X_selected)
Y = logreg.predict(X_selected) 
      


