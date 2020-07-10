#!/usr/bin/python2.4
 
# -*- coding: utf-8 -*-
 
 
"""Simple command-line example for Custom Search.
 

 
Command-line application that does a search.
 
"""
 
#**************************************************************
#global variables
__author__ = '[author name]'

 
key='[key]'
engine_id = '[id]'


import pprint
import csv
import json
import datetime
import sys
import codecs
import os


#amended for python 3.x
from googleapiclient.discovery import build

#specify 
sys.stdout = codecs.getwriter("iso-8859-1")(sys.stdout, 'xmlcharrefreplace')

#remove double quote for filename printing
def dequote(S):
  S=S.replace("\"","_")
  S=S.replace("\'","_")    
  return S

#**************************************************************
#this function print out the response objects into the textfile 
def printResponse(res,text_file,setCnt,resultPrinted):

  #TypeError: the JSON object must be str, not 'dict'
  x = json.dumps(res)
    
  searchTitle = res['queries']['request'][0]['title']
  total_results = res['searchInformation']['totalResults']
  total_results_int = int(total_results)

  if (total_results_int - resultPrinted) > 10:
    listsize = 10
  else:
    listsize = total_results_int - resultPrinted


  text_file.writelines(searchTitle)
  systemNow = datetime.datetime.now()
  systemNowStr = '{:%Y-%m-%d %H:%M}'.format(systemNow)
  text_file.writelines('Search done at '+ systemNowStr)
  text_file.writelines(' Returned record = '+total_results+'\n')

  if listsize > 0:
    res_snippet = [' '] * listsize
    res_link = [' '] * listsize
    res_title = [' '] * listsize

  i = 0
  if listsize >0:
    while i < listsize:
      res_snippet[i] = res["items"][i]["snippet"]
      res_link[i] = res["items"][i]["link"]
      res_title[i] = res["items"][i]["title"]
      resultPrinted = resultPrinted + 1

      text_file.write('\n')
      text_file.write('Result '+str(resultPrinted)+': ')
      text_file.write(res_title[i]+"\n")
      text_file.write(res_snippet[i]+"\n")
      text_file.write(res_link[i]+"\n")
      i = i + 1
  
  return resultPrinted

  ##  with open(fileNameCsv,'w',newline='') as fp:
  ##    f = csv.write(fp,delimiter=',')
  
  #f = csv.writer(open(fileNameCsv,"wb+"))
  #f.writerows(total_results)
  
  #text_file.write('original JSON\n'+x)

#**************************************************************
#this function interface with google custom search API, and passing the response result and textfile 
def gSearch(qName,outputF):
  service = build("customsearch", "v1",
            developerKey="AIzaSyBH5zIJfxDoHtdu8Sq1V0XumlxgoernOGg")

  #No. of result printed in files
  resultPrinted = 0
  
  #need 3 pages of results
  #search engine is set as UTF-8
  #p1
  res = service.cse().list(
    q= qName,
    cx='009533075362033359628:abwpaiibdgq',
    #lr = lang_en,
    num = 10,
    ).execute()
  #p2
  next_response = service.cse().list(
    q=qName,
    cx='009533075362033359628:abwpaiibdgq',
    #lr = lang_en,
    num=10,
    start=11,
    ).execute()
  #p3
  the_next_response = service.cse().list(
    q=qName,
    cx='009533075362033359628:abwpaiibdgq',
    #lr = lang_en,
    num=10,
    start=21,
    ).execute()
  
  
  pathname = os.path.dirname(os.path.abspath(__file__))
  fileNameCsv = pathname+"\output" + dequote(qName) + outputF + ".txt"
  text_file = open(fileNameCsv, "w",encoding = 'utf-8')
  total_results = res['searchInformation']['totalResults']
  total_results_cnt = int(total_results)

  #depends how much result returned, no. of pages to be shown
  if total_results_cnt >= 0:
    resultPrinted = printResponse(res,text_file,total_results_cnt,resultPrinted)
    if total_results_cnt >10:
      resultPrinted = printResponse(next_response,text_file,total_results_cnt,resultPrinted)
      if total_results_cnt >20:
        resultPrinted = printResponse(the_next_response,text_file,total_results_cnt,resultPrinted)

  text_file.close()
  
  #fileName = "c:\jeffrey ng\python\output" + dequote(qName) + "JSON.txt"

  #x = json.dumps(res)
  
  #with open(fileName, "w",encoding = 'utf-8') as text_file:
    #print(res, file=text_file)
#**************************************************************
def addDoubleQuote(theStr):
  theStr = '\"'+theStr +'\"'
  return theStr
#**************************************************************
#this function generates different google string search 
def qGenerator(qName,qEntityName,qKeyword,searchCnter):
  
  #name
  qName_Keywords = qName + addDoubleQuote(qEntityName) + addDoubleQuote(qKeyword)
  searchCnter  = searchCnter + 1
  gSearch(qName_Keywords,str(searchCnter))
  #double quoted name
  qName_Keywords = addDoubleQuote(qName) + addDoubleQuote(qEntityName) + addDoubleQuote(qKeyword)
  searchCnter  = searchCnter + 1
  gSearch(qName_Keywords,str(searchCnter))
  return searchCnter

#************************************************************** 
def main():
 
  # Build a service object for interacting with the API. Visit
  # the Google APIs Console <http://code.google.com/apis/console>
  # to get an API key for your own application.
  #try to read from input file and pass to qGenerator to process
  pathname = os.path.dirname(os.path.abspath(__file__))
  inputFileName = pathname+"\inputName.txt"
  inputFileEntity = pathname+"\InputEntityName.txt"
  inputKeyword = pathname+"\InputKeywords.txt"

  searchCnter = 0
  #Generate search
  for individual in open(inputFileName, "r",encoding = 'utf-8'):
    searchCnter = qGenerator(individual.rstrip(),'','',searchCnter)
    for entity in  open(inputFileEntity, "r",encoding = 'utf-8'):
        searchCnter = qGenerator(individual.rstrip(),entity.rstrip(),'',searchCnter)
        for keyword in  open(inputKeyword, "r",encoding = 'utf-8'):
          searchCnter = qGenerator(individual.rstrip(),entity.rstrip(),keyword.rstrip(),searchCnter)

if __name__ == '__main__':
  main()
 
