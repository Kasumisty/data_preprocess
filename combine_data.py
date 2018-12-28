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

# print(efiles)
# print('../data/ace_2005_td_v7/data/Chinese/bn/adj/CNR20001121.1700.1232.apf.xml' in file_dir)
# exit()
# print(len(file_dir))

file_dir = file_dir - efiles

# print(len(file_dir))
# exit()

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
