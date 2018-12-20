import re
import jieba
import pickle
from bs4 import BeautifulSoup
from my_code.utils import parseDirs, loadStopWords

types = ['bn', 'nw', 'wl']
tags = {'bn': 'turn', 'nw': 'text', 'wl': 'post'}
stopWordPath = 'stopwords.txt'
searchPattern = '.sgm'

SAVE_PATH = '../processed_data/segSentences.pkl'
DIR = ['../data/ace_2005_td_v7/data/Chinese/bn/adj',
       '../data/ace_2005_td_v7/data/Chinese/nw/adj',
       '../data/ace_2005_td_v7/data/Chinese/wl/adj']
files_dir = parseDirs(DIR, searchPattern)

# load stopwords
stopWords = loadStopWords(stopWordPath)

seg_sentences = []
for one in files_dir:
    with open(one, 'r') as f:
        soup = BeautifulSoup(f.read(), 'lxml')

    for type in types:
        if type in one:
            sentence = ''.join([element.get_text().replace('\n', '').replace(' ', '')
                                for element in soup.find_all(tags[type])])
    seg_sentence = []
    seg_words = list(jieba.cut(sentence))
    for seg in seg_words:
        if seg not in stopWords and len(seg) > 1 and not re.search(r'\d+\.?\d*', seg):
            seg_sentence.append(seg)
    seg_sentences.append(seg_sentence)

with open(SAVE_PATH, 'wb') as f:
    pickle.dump(seg_sentences, f)
