import os
from bs4 import BeautifulSoup
from my_code.utils import parseDirs

err_files_dir = '../processed_data/viewdata/datacheck.txt'
savePath = '../processed_data/combination.xml'
DIR = ['../data/ace_2005_td_v7/data/Chinese/bn/adj',
       '../data/ace_2005_td_v7/data/Chinese/nw/adj',
       '../data/ace_2005_td_v7/data/Chinese/wl/adj']
searchPattern = '.apf.xml'

def getErrFiles(file):
    with open(file, 'r') as f:
        return [line.strip() for line in f if line.startswith('.')]

file_dir = set(parseDirs(DIR, searchPattern))
efiles = set(getErrFiles(err_files_dir))

file_dir = file_dir - efiles

fullData = '<?xml version="1.0"?>\n<document>\n'
for one in file_dir:
    with open(one, 'r') as f:
        soup = BeautifulSoup(f.read(), 'lxml')
    events = soup.find_all('event')
    tmp = ''.join([str(event.prettify()) for event in events])
    base = os.path.basename(one)
    fullData += '<doc id=' + os.path.splitext(os.path.splitext(base)[0])[0] + '>\n'
    fullData += tmp
    fullData += '</doc>'
fullData += '</document>'

with open(savePath, 'w') as f:
    f.write(fullData)