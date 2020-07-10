'''*************************************
#1. Import libraries and key variable values

'''
from searchtweets import ResultStream, gen_rule_payload, load_credentials
from searchtweets import collect_results
import json
import os

script_dir = os.path.dirname(__file__)
#Twitter search commerical accounts credential
premium_search_args = load_credentials("~/.twitter_keys.yaml",
                                      env_overwrite=False)
#place the ".twitter_keys.yaml" under your home directory
#format and instruction on how please reference "twitter key setup.docx"

MAX_RESULTS=500 #maximum at 500

#list of companies in the same industry
comp_list = [line.rstrip('\n') for line in open('peer.csv')]

'''*************************************
#2. download tweets of each company

'''
for comp in comp_list:
    comp = '"'+comp +'"'
    rel_path = 'twitter'
    filename = comp+'data.json'
    json_abs_file_path = os.path.join(script_dir,rel_path,filename)

    filter_rule = comp

    rule = gen_rule_payload(filter_rule, results_per_call=100)
    print(rule)

    tweets = collect_results(rule,max_results=MAX_RESULTS,
                             result_stream_args=premium_search_args)

    with open(json_abs_file_path, 'w') as outfile:
        json.dump(tweets, outfile)

    filename = "header"+filter_rule+'.txt'
    text_abs_file_path = os.path.join(script_dir,rel_path,filename)
    f = open(text_abs_file_path,'w+')

    for tweet in tweets:
        f.write(tweet.all_text+'\t'+str(tweet.created_at_datetime)+'\n')
    f.close()
    
