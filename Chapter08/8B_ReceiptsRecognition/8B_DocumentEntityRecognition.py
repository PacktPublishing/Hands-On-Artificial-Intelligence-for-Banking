#1: Import relevant libriares and define variables
import os
import pandas as pd
from numpy import genfromtxt
import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
import gensim.downloader as api
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import preprocess_string, strip_tags, remove_stopwords,strip_numeric,strip_multiple_whitespaces
from scipy import linalg as LA
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,roc_curve, auc,confusion_matrix,f1_score

#please run this in terminal:  sudo apt-get install libopenblas-dev
model_word2vec = api.load("text8")  # load pre-trained words vectors

path = '/home/jeff/dataset/ghega-dataset/datasheets/'
list_of_dir = os.listdir(path)

truth_file_ext = '.groundtruth.csv'
blocks_file_ext = '.blocks.csv'
len_truth_file_ext = len(truth_file_ext)

png_file = '.out.000.png'
cnt_file = 0

re_sub = re.compile(r'[+|-]')

#2. Define functions relevant for works
##2A Neural Network
##2A_i. Grid search that simulate the performance of different neural network design
def grid_search(X_train,X_test, Y_train,Y_test,num_training_sample):
    
    best_f1 = 0
    best_hidden_layers_list = []
    best_hidden_layers_tuple = ()
    #various depth
    for depth in range(1,5):
        print('Depth = '+str(depth))
        for layer_size in range(1,8):
            neuron_cnt = 0
            hidden_layers_list = []        
            i = 0
            while i<depth:
                hidden_layers_list.append(layer_size)
                neuron_cnt += layer_size
                i+=1
            #pruning - to avoid over-training
            if num_training_sample<neuron_cnt:
                break
            
            hidden_layers_tuple = tuple(hidden_layers_list)
            nn_clf = MLPClassifier(alpha=1e-5,
                     hidden_layer_sizes=hidden_layers_tuple, random_state=1)
            
            nn_clf.fit(X_train,Y_train)
            Y_pred = nn_clf.predict(X_test)
            temp_f1 = f1_score(Y_test, Y_pred, average='samples')
            if temp_f1 > best_f1:
                best_f1 = temp_f1
                best_hidden_layers_list = hidden_layers_list
                best_hidden_layers_tuple = hidden_layers_tuple
    print(best_hidden_layers_list)
    return best_hidden_layers_list,best_hidden_layers_tuple
                
    #various size
# referencing: https://www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/
##2A_ii train network network
def train_NN(X,Y,target_names):
    print('Neural Network')
    #split the dataset into training set and testing set
    X[np.isnan(X)] = 0
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33, random_state=0)
    print('training set')
    print(X_train.shape)
    
    #preprocessing the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    num_training_sample = len(X_train)
    best_hidden_layers_list,best_hidden_layers_tuple = grid_search(X_train, X_test, Y_train, Y_test,num_training_sample)
    
    nn_clf = MLPClassifier(alpha=1e-5,
                     hidden_layer_sizes=best_hidden_layers_tuple, random_state=1)
    
    #fit the training data to the model
    nn_clf.fit(X_train,Y_train)
    Y_pred = nn_clf.predict(X_test)

    #common standard to compare across models
    print('f1')
    f1_clf = f1_score(Y_test, Y_pred, average='samples')
    print(f1_clf)
    print('classification report')
    print(classification_report(Y_test,Y_pred))

    ##save model
    f_nn = open('nn_clf.pkl',"wb+")
    pickle.dump(nn_clf, f_nn)
    f_nn.close()

    f_nn_sc = open('nn_scaler.pkl',"wb+")
    pickle.dump(scaler, f_nn_sc)
    f_nn_sc.close()

    '''
                  precision    recall  f1-score   support

               0       0.00      0.00      0.00        28
               1       1.00      0.12      0.22        16
               2       0.00      0.00      0.00        39
               3       0.00      0.00      0.00        31
               4       0.00      0.00      0.00        27
               5       0.60      0.52      0.56        29
               6       0.00      0.00      0.00        23
               7       0.00      0.00      0.00        19

       micro avg       0.63      0.08      0.14       212
       macro avg       0.20      0.08      0.10       212
    weighted avg       0.16      0.08      0.09       212
     samples avg       0.00      0.00      0.00       212

     ['Case', 'Model', 'PowerDissipation', 'StorageTemperature', 'ThermalResistance', 'Type', 'Voltage', 'Weigth']
    '''
    
    return nn_clf, f1_clf
#2B: prepare the text data series into numeric data series
#2B.i: cleanse text by removing multiple whitespaces and converting to lower cases
def cleanse_text(sentence,re_sub):
    #cleanse the sentence (stop word and lower case)
    #sentence = str(sentence)
    try:
        sentence = re_sub.sub(sentence,' sign')
    except Exception:
        sentence = '_'
    #print(sentence)
    CUSTOM_FILTERS = [lambda x: x.lower(),strip_multiple_whitespaces]
    #sentence_cleansed = remove_stopwords(sentence)
    sentence_cleansed = preprocess_string(sentence, CUSTOM_FILTERS)
    
    return sentence_cleansed

#2B.ii: convert text to numeric numbers
def text_series_to_np(txt_series,model,re_sub):
    len_array = np.zeros((len(txt_series),1))
    curr_row = 0
    for txt in txt_series:
        txt=str(txt)
        txt_cleansed = cleanse_text(txt,re_sub)
        txt_str = ' '.join(txt_cleansed)
        #print(txt_str)
        
        try:
            txt_vec = model(txt_str)
            txt_val = LA(txt_vec)
        except Exception:
            txt_val = 0
        len_array[curr_row][0] = txt_val
    return len_array
    
truth_names = ['element_type','page_label','x_label','y_label','w_label','h_label','text_label','page_value','x_value','y_value','w_value','h_value','text']
blocks_names = ['type','page','x','y','w','h','text','useless']
num_names = ['x','y','w','h']

#3. Loop through the files to prepare the dataset for training and testing
#loop through folders (represent different sources)
for folder in list_of_dir:
    files = os.path.join(path,folder)
    #loop through folders (represent different filing of the same source)
    for file in os.listdir(files):
        if file.endswith(truth_file_ext):
            #define the file names to be read
            print(folder + '/' +file)
            document_id = file[:-len_truth_file_ext]
            blocks_file = document_id+blocks_file_ext
            truth_file_path = os.path.join(path,folder,file)
            blocks_file_path = os.path.join(path,folder,blocks_file)

            #merge ground truth (aka target variables) with the blocks          
            f_df_truth = pd.read_csv(truth_file_path,quotechar='"',header=None,names=truth_names)
            f_df_blocks = pd.read_csv(blocks_file_path,quotechar='"',header=None,names=blocks_names)
            f_df = f_df_blocks.iloc[:,2:-1].merge(f_df_truth,on='text',how='outer')

            num_df = f_df[num_names].values
            
            #convert the text itself into vectors and lastly a single value using Eigenvalue
            text_df = f_df['text']
            text_np = text_series_to_np(text_df,model_word2vec,re_sub)

            label_df = f_df['text_label']
            label_np = text_series_to_np(label_df,model_word2vec,re_sub)

            target_df = f_df['element_type']

            X_np = np.hstack((num_df,text_np))
            
            if cnt_file ==0:
                           full_X_np = X_np
                           targets_df = target_df
            else:
                           full_X_np = np.vstack((full_X_np,X_np))
                           targets_df = targets_df.append(target_df)

                           
            cnt_file += 1
Y_pd = pd.get_dummies(targets_df)
Y_np = Y_pd.values
dummy_header = list(Y_pd.columns.values)
print('Dummy header: from 0 to 7')
print(dummy_header)

#4. Execute the training and test the outcome
NN_clf, f1_clf = train_NN(full_X_np,Y_np,dummy_header)
np.save('Y_np',Y_np)
np.save('X_np',full_X_np)
