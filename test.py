# from gensim.models import Word2Vec
#
# MODEL_FILE = '../processed_data/wordVec_150d'
# model = Word2Vec.load(MODEL_FILE)
# a = model.similarity('中国', '台湾')
# b = model.similarity('总统', '主席')
# c = model.most_similar('中国', topn=20)
# d = model.most_similar('总统', topn=20)
# print(a)
# print(b)
# print(c)
# print(d)


# from bs4 import BeautifulSoup
#
# # FILE_PATH = '../data/ace_2005_td_v7/data/Chinese/bn/adj/CBS20001006.1000.0074.apf.xml'
# # FILE_PATH2 = '../data/ace_2005_td_v7/data/Chinese/bn/adj/CBS20001006.1000.0074.sgm'
#
# FILE_PATH = '../data/ace_2005_td_v7/data/Chinese/wl/adj/DAVYZW_20041223.1020.ag.xml'
# FILE_PATH2= '../data/ace_2005_td_v7/data/Chinese/wl/adj/DAVYZW_20041223.1020.sgm'
#
# with open(FILE_PATH2, 'r') as f:
#     data = f.read()
#     soup = BeautifulSoup(data, 'lxml')
# context = soup.get_text()
# print(context)
#
# with open(FILE_PATH, 'r') as f:
#     soup = BeautifulSoup(f.read(), 'lxml')
# chars = soup.find_all('charseq')
# for char in chars:
#     start = int(char.get('start')) -1
#     # print(start, type(start))
#     end = int(char.get('end'))
#     # print(end, type(end))
#     sgm = context[start: end]#.replace(r'\n', ' ')
#     # print('apf', char.get_text())
#     # print('seq', sgm)
#     # print()
#     if sgm != char.get_text():
#         print('apf', char.get_text(), start)
#         print('sgm', sgm)


# import pickle
# with open('../processed_data/etype_with_arguments.pkl', 'rb') as f:
#     print(pickle.load(f))

# count number of each event
# from bs4 import BeautifulSoup
# with open('../processed_data/combination.xml', 'r') as f:
#     soup = BeautifulSoup(f.read(), 'lxml')
# events = soup.find_all('event')
# count = {}
# for event in events:
#     subtype = event.get('subtype')
#     num = len(event.find_all('event_mention'))
#     if subtype in count:
#         count[subtype] += num
#     else:
#         count[subtype] = num
# for type in count:
#     print(type + ',' + str(count[type]))

# data = ''
# with open('../processed_data/EventType.md', 'r') as f:
#     for line in f:
#         line = line.strip()
#         if line.startswith('#'):
#             data += '\n' + line[2:] + ' '
#         if line.startswith('*'):
#             data +=  line[2:] + ' '
# print(data)


# import re
# s = re.split(r'[?,.]', 'dsa,456.89?1323')
# print(s)
#
# import operator
# from functools import reduce
# a = [[1,2,3], [4,6], [7,8,9,8]]
# print(reduce(lambda x,y: x+y, a))
#
#
# # 阶乘
# x = reduce(operator.mul, range(1,4))


# import collections
# from bs4 import BeautifulSoup
#
# with open('../data/ace_2005_td_v7/data/Chinese/bn/adj/CNR20001103.1700.0856.sgm') as f:
#     soup = BeautifulSoup(f.read(), 'lxml')
#     text = soup.get_text()
#
# seqDict = {}
# orderDict = collections.OrderedDict()
#
# with open('../data/ace_2005_td_v7/data/Chinese/bn/adj/CNR20001103.1700.0856.apf.xml', 'r') as f:
#     soup = BeautifulSoup(f.read(), 'lxml')
# charseqs = soup.find_all('charseq')
# for charseq in charseqs:
#     start = int(charseq.get('start'))
#     seqDict[start] = charseq
#
# for key in sorted(seqDict):
#     orderDict[key] = seqDict[key]
#
# for start in orderDict:
#     seq = orderDict[start].get_text()
#     end = int(orderDict[start].get('end'))
#     sgm_text = text[start-3: end-2]     # [start-3: end-2] [start-1: end]
#     if not seq == sgm_text:
#         print(seq)
#         print(sgm_text)
#         print()


# import jieba
# str = '''被怀疑弃保潜逃的中央电台前董事长朱婉清目前是从洛杉矶跑到东岸的纽约 '''
# seg = list(jieba.cut(str))
# for s in seg:
#     print(s)

file = '../processed_data/predata_one_mention.txt'
with open(file, 'r', encoding='utf-8') as f:
    data = f.readlines()

# for line in data:
#     if line == '\n':
#         print('true')
# exit()
tag = 1
count = 0
all = 0
for line in data:
    if line == '\n':
        # print('true')
        all += 1
        if tag:
            print('err')
            count += 1
        tag = 1
    if len(line.split('\t')) > 1:
        tag = 0
print(count)
print(all)
