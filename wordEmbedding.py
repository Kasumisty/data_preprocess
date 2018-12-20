import pickle
from gensim.models import Word2Vec

FILE_PATH = '../processed_data/segSentences.pkl'
SAVE_PATH = '../processed_data/wordVec_150d'
with open(FILE_PATH, 'rb') as f:
    sentence = pickle.load(f)

model = Word2Vec(sentence, size=150)
model.save(SAVE_PATH)
