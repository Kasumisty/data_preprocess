from gensim.models import Word2Vec

vec_file = '../processed_data/wordVec_150d'
file = '../processed_data/predata.txt'

word2Vec = Word2Vec.load(vec_file)

with open(file, 'r') as f:
    for line in f:
        if line.startswith('i'):
            continue
    # todo
