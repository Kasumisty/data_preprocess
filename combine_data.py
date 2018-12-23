from bs4 import BeautifulSoup
from my_code.utils import parseDirs

savePath = '../processed_data/combination.xml'
DIR = ['../data/ace_2005_td_v7/data/Chinese/bn/adj',
       '../data/ace_2005_td_v7/data/Chinese/nw/adj',
       '../data/ace_2005_td_v7/data/Chinese/wl/adj']

searchPattern = '.apf.xml'
file_dir = parseDirs(DIR, searchPattern)

fullData = '<?xml version="1.0"?>\n<document>\n'
for one in file_dir:
    with open(one, 'r') as f:
        soup = BeautifulSoup(f.read(), 'lxml')
    events = soup.find_all('event')
    tmp = ''.join([str(event.prettify()) for event in events])
    fullData += tmp
fullData += '</document>'

with open(savePath, 'w') as f:
    f.write(fullData)
