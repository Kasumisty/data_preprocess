import pickle
from gensim.models import Word2Vec

SIZE = 150
FILE_PATH = '../processed_data/segSentences.pkl'
SAVE_PATH = '../processed_data/wordVec_' + str(SIZE) + 'd'
with open(FILE_PATH, 'rb') as f:
    sentence = pickle.load(f)

model = Word2Vec(sentence, size=SIZE)
model.save(SAVE_PATH)