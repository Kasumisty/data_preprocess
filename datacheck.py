import re
import operator
from functools import reduce
from bs4 import BeautifulSoup
from my_code.utils import parseDirs

DIR = [
    # '../data/ace_2005_td_v7/data/Chinese/bn/adj',
    #    '../data/ace_2005_td_v7/data/Chinese/nw/adj',
    '../data/ace_2005_td_v7/data/Chinese/wl/adj'
]
# TIME_PATTERN = r'\d{2}/\d{2}/\d{4}'
TIME_PATTERN = r'\d{4}-\d{1,2}-\d{1,2}'

file_dir = parseDirs(DIR, '.apf.xml')

err_files = set()

for one in file_dir:
    sgm_file = one.replace('.apf.xml', '.sgm')
    with open(sgm_file, 'r', encoding='utf-8', newline='\r\n') as f:
        soup = BeautifulSoup(f.read(), 'lxml')
        text = soup.get_text()

    with open(one, 'r', encoding='utf-8', newline='\r\n') as f:
        soup = BeautifulSoup(f.read(), 'lxml')
    charseqs = soup.find_all('charseq')
    for charseq in charseqs:
        start = int(charseq.get('start')) - 1
        end = int(charseq.get('end'))
        seq = charseq.get_text()
        sgm_text = text[start:end]

        if seq != sgm_text:
            if sgm_file not in err_files and not re.search(TIME_PATTERN, seq):
                err_files.add(sgm_file)
                print(sgm_file)
            print(seq)
            print(sgm_text)
            print()

for i in err_files:
    print(i)
print(len(file_dir))
