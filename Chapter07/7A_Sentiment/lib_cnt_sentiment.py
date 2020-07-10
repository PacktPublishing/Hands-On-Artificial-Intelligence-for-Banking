import os

#generate text given url
import requests
from bs4 import BeautifulSoup
import re
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()



'''
Dataset from Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews." Proceedings of the ACM SIGKDD International Conference on Knowledge, Discovery and Data Mining (KDD-2004), Aug 22-25, 2004, Seattle,Washington, USA, 

there have been some invalid characters in the negative words file, i open it and save it as utf to correcct the issue
'''
script_dir = os.path.dirname(__file__)
pos_abs_file_path = os.path.join(script_dir,'lexicon/positive-words.txt')
f_pos_list = open(pos_abs_file_path,'r')
pos_list = [line.rstrip('\n') for line in f_pos_list.readlines()]

neg_abs_file_path = os.path.join(script_dir,'lexicon/negative-words.txt')
f_neg_list = open(neg_abs_file_path,'r')
neg_list = [line.rstrip('\n') for line in f_neg_list.readlines()]

#cal the positive and negative sentiment words given the text
def cnt_sentiment(text_to_be_parsed):
    pos_s = 0
    neg_s = 0
    words = text_to_be_parsed.lower().split()
    for word in words:
        if word in pos_list:
            pos_s+=1
        if word in neg_list:
            neg_s+=1
    return pos_s, neg_s

#Generate a list of sentence given the sentences
def prep_sentence_list(spacy_sentences):
    list_sentences= []
    for sentence in spacy_sentences:
        list_sentences.append(str(sentence).lower().split())
        
    return list_sentences
'''
from gensim.models import Word2Vec
#topic model
def train_Word2Vec_model(sentences,file_name = 'model.bin',new_model = True):
    if new_model:
        # train model
        model = Word2Vec(sentences, min_count=1)
        # save model
        model.save(file_name)
    else:
        model = Word2Vec.load(file_name)
    # summarize the loaded model
    print(model)
    # summarize vocabulary
    words = list(model.wv.vocab)
    
    
    return model
'''

def noun_phrase(sentence,item_list,lower):
    #doc = nlp_en(phrase) # create spacy object
    #doc = nlp(phrase)
    #token_not_noun = []
    noun_list = []
    noun = ''
    #if len(item_list)>0:
    #    add_item = True
    #else:
    #    add_item = False

    for item in sentence:
        if item.pos_ != "NOUN": # separate nouns and not nouns
            if len(noun) > 0:
                if lower:
                    noun = noun.lower()
                noun_list.append(noun)
                item_list.append(noun)
                noun = ''
            
        if item.pos_ == "NOUN":
            if len(noun) == 0:
                noun = item.text
            else:
                noun += ' ' + item.text


    #handle the last word that might be a noun
    if len(noun) > 0:
        if lower:
            noun = noun.lower()
        noun_list.append(noun)
        item_list.append(noun)
        noun=''

    return noun_list, item_list


#NER
import spacy
from spacy import displacy
from collections import Counter
import math

#text has to be less than 1000000
def NER_topics(text_to_be_parsed):

    #POS_list = ['PROPN','VERB','NOUN']
    words_list=[]
    items_list=[]
    MAX_SIZE =100000
    txt_len = len(text_to_be_parsed)
    number_nlp = math.ceil(txt_len/MAX_SIZE)
    full_sentences_list=[]

    for nlp_cnt in range(number_nlp):
        start_pos = nlp_cnt*MAX_SIZE
        end_pos = min(MAX_SIZE,txt_len-start_pos)+start_pos-1
        txt_selected = text_to_be_parsed[start_pos:end_pos]
    
        article = nlp(txt_selected)

        #lemmize the text
        #items = [x.text for x in article]
        
        #if not x.is_stop and x.Pos_ != 'PUNCT'
        sentences_list = [x for x in article.sents]
        full_sentences_list+=sentences_list
        for sent in sentences_list:
            phrases_list =[]
            phases_list,items_list = noun_phrase(sent,items_list,lower=True)
        
        
    common_words = Counter(items_list).most_common(50)
    for word in common_words:
        words_list.append(word[0])
    
    return common_words,full_sentences_list,words_list

'''
str_test = 'We assume that readers you have already done some works to study the businesesbusiness problem before focusing on this particular example of application\n. Fs;\n whereas for data understanding, as in real life, data will be understood as and when we went go through the test of fires through various stages of raising requests and clarifying requirements with IT /system owners that write the script to output the data.\n'
_,_,str_temp = NER(str(str_test))

article = nlp(str_test)

#lemmize the text
items = [x.text for x in article.ents]
common_words = Counter(items).most_common(50)
#if not x.is_stop and x.Pos_ != 'PUNCT'
sentences = [x for x in article.sents]
'''

#convert the URL's content into string
def url_to_string(url):
    try:
        res = requests.get(url)
    except Exception:
        return " "
    html = res.text
    soup = BeautifulSoup(html, 'html5lib')
    try:
        soup_content = soup(["script", "style", 'aside'])
    except Exception:
        return ""
    for script in soup_content:
        script.extract()
    #all_strings = [e for e in soup.recursiveChildGenerator() 
    #     if isinstance(e,str)]
    return " ".join(re.split(r'[\n\t]+', soup.get_text()))

'''
ny_bb = url_to_string('https://www.nytimes.com/2018/08/13/us/politics/peter-strzok-fired-fbi.html?hp&action=click&pgtype=Homepage&clickSource=story-heading&module=first-column-region&region=top-news&WT.nav=top-news')
article = nlp(ny_bb)
len(article.ents)
'''
