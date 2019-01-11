import numpy as np
from collections import defaultdict
from gensim.models import Word2Vec


# model = Word2Vec.load(vec_file)
# if '发行' in model:
#     print('123456')
# todo
# trigger 不在词向量中
def lookup(mess, key, dic):
    if key not in dic:
        dic[key] = len(dic)
        print(mess + ':', key, '==>', dic[key])


def parseInst(inst, model, eventTypeDict, maxlen=100, k=150):
    triggerInfo = []
    embeddings = np.zeros(shape=[maxlen, k])
    count = 0
    for line in inst:
        line = line.split('\t')
        if len(line) > 1:
            lookup('eventType', line[1], eventTypeDict)
            if line[0] not in model:
                print('err', line[0])
            else:
                embeddings[count] = np.array(model[line[0]])
                # print(line[0])
            triggerInfo.append((count, eventTypeDict[line[1]]))
            count += 1
        else:
            if line[0] in model:
                embeddings[count] = np.array(model[line[0]])
                # print(line[0])
                count += 1
    return count, embeddings, triggerInfo


# def getEmbedding(model, inst, candidate=None, maxlen=32, k=150):
#     embeddings = np.zeros(shape=[maxlen, k])
#     count = 0
#     for word in inst:
#         if word in model:
#             try:
#                 embeddings[count] = np.array(model[word])
#                 count += 1
#             except OverflowError:
#                 pass
#     return count, embeddings

if __name__ == '__main__':
    vec_file = '../processed_data/wordVec_150d'
    file = '../processed_data/predata.txt'

    model = Word2Vec.load(vec_file)
    eventTypeDict = defaultdict(int)
    maxlen = 0
    revs = []
    inst = []
    # c, e = getEmbedding(model, ['中国', ',', '中央'])
    # print(c, e)
    # exit()
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('i'):
                continue
            if line:
                inst += [line]
                continue
            # print(inst)
            # break
            instLen, embeddings, triggerInfo = parseInst(inst, model, eventTypeDict, maxlen=42)
            # print(triggerInfo)
            # print(eventTypeDict)
            # print(instLen)
            # print(embeddings[:instLen])
            # break
            if instLen > maxlen:
                maxlen = instLen
            # break
            revs.append([embeddings, triggerInfo, instLen])
            inst = []
    print(maxlen)
