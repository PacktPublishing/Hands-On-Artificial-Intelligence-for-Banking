import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

bank_file_name = './dataset/bank-full.csv'
df_bank = pd.read_csv(bank_file_name,sep=";")

census_file_name = 'CENSUS_DATA.csv'
df_xtics = pd.read_csv(census_file_name,quotechar='"')

#df_bank[['age']].values
cat_val = ''
cat_dict = {}
for index, row in df_xtics.iterrows():
    curr_cat = row['f_Characteristic']
    curr_cat_val = row['Characteristic']
    if curr_cat_val != cat_val:
        if curr_cat == curr_cat_val:
            cat_dict[curr_cat]={}
        else:
            cat_dict[curr_cat][curr_cat_val]=row['f_Interest Earning ']


df_bank['age_c'] = pd.cut(df_bank['age'], [0,35,45,55,65,70,75,200])
'''
pd.unique(df_bank['age_c'].values)
pd.unique(df_bank['age_c'].values)
[(55, 65], (35, 45], (0, 35], (45, 55], (65, 70], (75, 200], (70, 75]]
Categories (7, interval[int64]): [(0, 35] < (35, 45] < (45, 55] < (55, 65] < (65, 70] < (70, 75] <
                                  (75, 200]]

cat_dict['Age of Householder']
{'35 to 44 years': -1800.0, '.70 to 74 years': 5200.0, '.75 and over': 11200.0, '65 years and over': 7100.0, '.65 to 69 years': 4931.0, '45 to 54 years': -800.0, '55 to 64 years': 1200.0
, 'Less than 35 years': -2800.0}

'''
#ID Conversions
df_bank['age_c_codes']=df_bank['age_c'].cat.codes.astype(str)
age_map={'0':'Less than 35 years'
,'1':'35 to 44 years'
,'2':'45 to 54 years'
,'3':'55 to 64 years'
,'4':'.65 to 69 years'
,'5':'.70 to 74 years'
,'6':'.75 and over'}

#map back the survey data
df_bank['age_c1']=df_bank['age_c_codes'].map(age_map)
df_bank['age_c1_val']=df_bank['age_c1'].map(cat_dict['Age of Householder'])

#df_bank[['age','age_c1']]
#df_bank[['age_c','age_c_codes']]

X_flds = ['balance','day', 'duration', 'pdays',
       'previous', 'age_c1_val']
X = df_bank[X_flds]
y = df_bank['y']
X, y = make_classification(n_samples=1000, n_features=3,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                             random_state=0)
clf.fit(X, y)
print(clf.feature_importances_)

#TODO:
#split training and testing set
#change binary fields to 1 or 0
#map the rest
