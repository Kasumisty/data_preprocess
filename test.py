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


# import numpy as np
# import tensorflow as tf
#
# x = np.array([1, 6, 2, 7, 3, 8, 4, 9, 5, 10, 1, 6, 2, 7, 3, 8, 4, 9, 5, 10, 1, 6, 2, 7, 3, 8, 4, 9, 5, 10]).reshape(
#     [3, 5, 1, 2])
# # print(x)
# X = tf.placeholder(tf.float32, shape=[3, 5, 1, 2])
# s = tf.unstack(x, axis=0)
# g = tf.unstack(s[0], axis=1)
# k = tf.unstack(g[0], axis=1)
# with tf.Session() as sess:
#     o1 = sess.run(s)
#     print(o1[0])
#
#     o2 = sess.run(g)
#     print(o2[0])
#
#     o3 = sess.run(k)
#     print(o3)


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


# file_path = '../processed_data/predata.txt'
# maxlen = 0
# with open(file_path, 'r') as f:
#     count = 0
#     for line in f:
#         if line.startswith('i'):
#             continue
#         if line == '\n':
#             if count > maxlen:
#                 maxlen = count
#             count = 0
#             continue
#         count += 1
# print(maxlen)


# from gensim.models import Word2Vec
# model = Word2Vec.load('../data/zh/zh.bin')
# a = model.similarity('中国', '台湾')
# b = model.similarity('总统', '主席')
# c = model.most_similar('中国', topn=20)
# d = model.most_similar('总统', topn=20)
# print(a)
# print(b)
# print(c)
# print(d)


# polyglot tutorial
# import pickle
# with open('../data/zh/polyglot-en.pkl', 'rb') as f:
#     words, embeddings = pickle.load(f, encoding='iso-8859-1')
# print("Emebddings shape is {}".format(embeddings.shape))
#
# """KNN Example."""
#
# from operator import itemgetter
# from itertools import islice
# import re
# import numpy
#
# # Special tokens
# Token_ID = {"<UNK>": 0, "<S>": 1, "</S>":2, "<PAD>": 3}
# ID_Token = {v:k for k,v in Token_ID.items()}
#
# # Map words to indices and vice versa
# word_id = {w:i for (i, w) in enumerate(words)}
# id_word = dict(enumerate(words))
#
# # Noramlize digits by replacing them with #
# DIGITS = re.compile("[0-9]", re.UNICODE)
#
# # Number of neighbors to return.
# k = 5
#
#
# def case_normalizer(word, dictionary):
#   """ In case the word is not available in the vocabulary,
#      we can try multiple case normalizing procedure.
#      We consider the best substitute to be the one with the lowest index,
#      which is equivalent to the most frequent alternative."""
#   w = word
#   lower = (dictionary.get(w.lower(), 1e12), w.lower())
#   upper = (dictionary.get(w.upper(), 1e12), w.upper())
#   title = (dictionary.get(w.title(), 1e12), w.title())
#   results = [lower, upper, title]
#   results.sort()
#   index, w = results[0]
#   if index != 1e12:
#     return w
#   return word
#
#
# def normalize(word, word_id):
#     """ Find the closest alternative in case the word is OOV."""
#     if not word in word_id:
#         word = DIGITS.sub("#", word)
#     if not word in word_id:
#         word = case_normalizer(word, word_id)
#
#     if not word in word_id:
#         return None
#     return word
#
#
# def l2_nearest(embeddings, word_index, k):
#     """Sorts words according to their Euclidean distance.
#        To use cosine distance, embeddings has to be normalized so that their l2 norm is 1."""
#
#     e = embeddings[word_index]
#     distances = (((embeddings - e) ** 2).sum(axis=1) ** 0.5)
#     sorted_distances = sorted(enumerate(distances), key=itemgetter(1))
#     return zip(*sorted_distances[:k])
#
#
# def knn(word, embeddings, word_id, id_word):
#     word = normalize(word, word_id)
#     if not word:
#         print("OOV word")
#         return
#     word_index = word_id[word]
#     indices, distances = l2_nearest(embeddings, word_index, k)
#     neighbors = [id_word[idx] for idx in indices]
#     for i, (word, distance) in enumerate(zip(neighbors, distances)):
#       print(i, '\t', word, '\t\t', distance)
#
# knn("Jordan", embeddings, word_id, id_word)
# print()
# knn("1986", embeddings, word_id, id_word)
# print()
# knn("JAPAN", embeddings, word_id, id_word)


# # -*- coding: utf-8 -*-
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder().fit([1,111,122,188,999])
# le_transform = le.transform([1,122,111])
# print(le_transform)


# import pandas as pd
# from sklearn import  preprocessing
#
# test=pd.DataFrame({'city':['beijing','shanghai','shenzhen'],'age':[21,33,23],'target':[0,1,0]})
#
# label = preprocessing.LabelEncoder()
# test['city']= label.fit_transform(test['city'])
# print(test)
#
# enc=preprocessing.OneHotEncoder(categorical_features=[1], sparse=False)
# test=enc.fit_transform(test)
# print(test)
#
# # print(pd.factorize(test['city']))
#
# print(pd.get_dummies(test['city'],prefix='city'))


# s = '爱一个人 如何厮守到老 怎样面对一切 我不知道'
# print(s.replace(' ', ''))

# for i, w in enumerate(s):
#     print(i, w)


# l = list(s)
# print(l)

import jieba

s = '送医不治'
print(list(jieba.cut(s)))
