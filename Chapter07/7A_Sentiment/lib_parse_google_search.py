from googleapiclient.discovery import build
import pprint
import json

#my_api_key = "Google API key"
my_api_key = '[api key]'
my_cse_id = '[cse id]'


'''
def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res['items']
'''
def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res['items']

import math
#download DUKE Energy
def parse_google_search(search_txt,save_path,pastDays,numSearch):
    #search_txt = 'DUKE Energy'
    all_results=[]
    total_number_search = math.ceil(numSearch/10)
    for search_num in range(total_number_search):
        print(search_num)
        start_num = (search_num+1*10)+1
        try:
            results = google_search(search_txt, my_api_key, my_cse_id, num=10,start=start_num,dateRestrict='d'+str(pastDays))
        except Exception:
            return all_results
        all_results+=results
    
    f = open(save_path, 'w+')
    json.dump(all_results,f)
    return all_results
'''
for result in results:
    pprint.pprint(result)
'''
#service = build('books', 'v1', developerKey="api_key")

