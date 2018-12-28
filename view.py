import pickle
import collections
from bs4 import BeautifulSoup
from my_code.utils import parseDirs

SAVE_PATH = '../processed_data/etype_with_arguments.pkl'
DIR = ['../data/ace_2005_td_v7/data/Chinese/bn/adj',
       '../data/ace_2005_td_v7/data/Chinese/nw/adj',
       '../data/ace_2005_td_v7/data/Chinese/wl/adj']
search_pattern = '.apf.xml'
files_dir = parseDirs(DIR, search_pattern)

typeDic = collections.defaultdict(set)
for one in files_dir:
    with open(one, 'r') as f:
        soup = BeautifulSoup(f.read(), 'lxml')
        events = soup.find_all('event')
        for event in events:
            etype = event.get('subtype')
            arguments = event.find_all('event_argument')
            typeDic[etype] |= set([arg.get('role') for arg in arguments])
with open(SAVE_PATH, 'wb') as f:
    pickle.dump(typeDic, f)
