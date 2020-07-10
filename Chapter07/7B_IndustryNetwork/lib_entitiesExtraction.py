import os
import spacy
import en_core_web_lg
nlp = en_core_web_lg.load()
#NER
import spacy
from spacy import displacy
from collections import Counter
import math


#convert the list of sentences to lower case
def prep_sentence_list(spacy_sentences):
    list_sentences= []
    for sentence in spacy_sentences:
        list_sentences.append(str(sentence).lower().split())
        
    return list_sentences


#add in verb extractions
def noun_phrase(sentence,item_list,lower):
    noun_list = []
    noun = ''
    verb_list=[]
    
    for item in sentence:
        if item.pos_ != "NOUN": # separate nouns and not nouns
            if len(noun) > 0:
                if lower:
                    noun = noun.lower()
                noun=noun.encode()
                noun=noun.strip()
                noun_list.append(noun)
                item_list.append(noun)
                if item.pos_ == "VERB":
                    verb=item.text.strip()
                    verb_list.append(verb)
                    verb=''
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
        noun = noun.encode().strip()
        noun_list.append(noun)
        item_list.append(noun)
        noun=''

    return noun_list, item_list, verb_list

#text has to be less than 1000000
def NER_topics(text_to_be_parsed,lower=True):

    #POS_list = ['PROPN','VERB','NOUN']
    words_list=[]
    verbs_list=[]
    items_list=[]
    MAX_SIZE =10000
    txt_len = len(text_to_be_parsed)
    number_nlp = math.ceil(txt_len/MAX_SIZE)
    full_sentences_list=[]

    for nlp_cnt in range(number_nlp):
        start_pos = nlp_cnt*MAX_SIZE
        end_pos = min(MAX_SIZE,txt_len-start_pos)+start_pos-1
        txt_selected = text_to_be_parsed[start_pos:end_pos]
        #print(len(txt_selected))
        article = nlp(txt_selected)

        #lemmize the text
        #items = [x.text for x in article]
        
        #if not x.is_stop and x.Pos_ != 'PUNCT'
        sentences_list = [x for x in article.sents]
        full_sentences_list+=sentences_list
        for sent in sentences_list:
            phrases_list =[]
            verb_list=[]
            phases_list,items_list,verb_list = noun_phrase(sent,items_list,lower=lower)
            verbs_list.append(verb_list)
        
        
    common_words = Counter(items_list).most_common(50)
    for word in common_words:
        words_list.append(word[0])
    
    return common_words,full_sentences_list,words_list,verbs_list

#extract organization
def org_extraction(text_to_be_parsed):
    MAX_SIZE =10000
    txt_len = len(text_to_be_parsed)
    number_nlp = math.ceil(txt_len/MAX_SIZE)
    full_sentences_list=[]

    for nlp_cnt in range(number_nlp):
        start_pos = nlp_cnt*MAX_SIZE
        end_pos = min(MAX_SIZE,txt_len-start_pos)+start_pos-1
        txt_selected = text_to_be_parsed[start_pos:end_pos]
    
        article = nlp(txt_selected)
        for sentence in article.sents:
            sentence_list= []
            for ent in sentence.ents:
        #for ent in article.ents:
                sentence_list.append([ent.text, ent.label_])
            full_sentences_list.append(sentence_list)
    return full_sentences_list
