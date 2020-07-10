import lib_parse_google_search as cse
import os

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 07:38:32 2018

@author: jeff
"""

#load tweet json
script_dir = os.path.dirname(__file__)
rel_path = 'search/data.json'
json_abs_file_path = os.path.join(script_dir,rel_path)


q = '"DUKE Energy"'
pastDays = 365*5
numSearch = 1000
results = cse.parse_google_search(q,json_abs_file_path,pastDays,numSearch)

