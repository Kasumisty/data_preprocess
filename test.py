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


# # try to find whether each sentebnce has a trigger
# file = '../processed_data/predata.txt'
# with open(file, 'r', encoding='utf-8') as f:
#     data = f.readlines()
#
# tag = 1
# count = 0
# all = 0
# for line in data:
#     if line == '\n':
#         # print('true')
#         all += 1
#         if tag:
#             print('err')
#             count += 1
#         tag = 1
#     if len(line.split('\t')) > 1:
#         tag = 0
# print(count)
# print(all)


import numpy as np
import tensorflow as tf

x = np.array([1, 6, 2, 7, 3, 8, 4, 9, 5, 10, 1, 6, 2, 7, 3, 8, 4, 9, 5, 10, 1, 6, 2, 7, 3, 8, 4, 9, 5, 10]).reshape(
    [3, 5, 1, 2])
# print(x)
X = tf.placeholder(tf.float32, shape=[3, 5, 1, 2])
s = tf.unstack(x, axis=0)
g = tf.unstack(s[0], axis=1)
k = tf.unstack(g[0], axis=1)
with tf.Session() as sess:
    o1 = sess.run(s)
    print(o1[0])

    o2 = sess.run(g)
    print(o2[0])

    o3 = sess.run(k)
    print(o3)
    # o3 =
# print(x[0,:,:,0])
# print()
# print(x[0,:,:,1])
# print()

# print(x)
# print()
# exit()

# X = tf.placeholder(tf.float32, shape=[3,5,1,2])
# def pooling(X):
#     s = np.array([]).astype(np.float32)
#     # return tf.reduce_max(X[0, :3, 0, 0])
#     for i in range(X.shape[0]):
#         for j in range(X.shape[-1]):
#             s = tf.concat([ s, [tf.reduce_max(X[i, :3, 0, j])] ], axis=0)
#         for j in range(X.shape[-1]):
#             s = tf.concat([ s, [tf.reduce_max(X[i, 3:5, 0, j])] ], axis=0)
#     return tf.reshape(s, [3,2,1,2])
#
# def f(Y):
#     s = np.array([])
#     for i in range(Y.shape[0]):
#         s = tf.concat([s, Y[i, :, 0, 0], Y[i, :, 0, 1]], axis=0)
#     return tf.reshape(s, [Y.shape[0], 4, 1, 1])
#
# conv = tf.layers.conv2d(X, 2, kernel_size=[2,2], padding='same')
# pool = pooling(conv)
# dmpool = f(pool)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     o = sess.run(pool, feed_dict={X: x})
#     print(o.shape)


# def f(Y):
#     data = np.array([])
#     for i in range(3):
#         s = np.concatenate([Y[i, :, 0, 0], Y[i, :, 0, 1]], axis=0)
#         print(s)
#         data = np.append(data, s)
#     data = data.reshape([3, 4, 1, 1])
#     return data
#     # print(data[0, :, 0, 0])
#
# o = f(Y)


# data = np.array([])
# for i in range(3):
#     s = np.concatenate([Y[i, :, 0, 0], Y[i, :, 0, 1]], axis=0)
#     print(s)
#     data = np.append(data, s)
# data = data.reshape([3,4,1,1])
# print(data[0, :,0,0])


# def maxpool(x):
#     data = []
#     for i in range(x.shape[0]):
#         for j in range(x.shape[-1]):
#             # tf.concat([x[i, 0:3, :, j]])
#             data.append(x[i, 0:3, :, j])
#     tf.reshape(data, [3, 4, 1])

# s = maxpool(x)


# c = [[[[ 1,  2]]
#
#   [[ 3,  4]]
#
#   [[ 5,  6]]
#
#   [[ 7,  8]]
#
#   [[ 9, 10]]]
#
#
#  [[[ 1,  2]]
#
#   [[ 3,  4]]
#
#   [[ 5,  6]]
#
#   [[ 7,  8]]
#
#   [[ 9, 10]]]
#
#
#  [[[ 1 , 2]]
#
#   [[ 3,  4]]
#
#   [[ 5 , 6]]
#
#   [[ 7,  8]]
#
#   [[ 9 ,10]]]]
