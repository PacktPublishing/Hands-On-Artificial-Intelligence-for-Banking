#this program provides the list of banks
from bs4 import BeautifulSoup

import requests

url = 'https://en.wikipedia.org/wiki/List_of_banks_(alphabetically)'
r = requests.get(url)

data = r.text
soup = BeautifulSoup(data,'lxml')

f = open('list_of_banks.csv','w+')

for cell in soup.find_all('li'):
    for link in cell.find_all('a'):
        bank_name = link.string
        if bank_name:
            f.write(bank_name+'\n')

f.close()
