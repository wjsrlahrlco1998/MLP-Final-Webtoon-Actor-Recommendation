# Package Load
from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity

# Define Class
class Text:
    def __init__(self, model_path='./model/fasttext.model'):
        '''Setting Model'''
        self.model = FastText.load(model_path)
    
    def get_keyword_similarity(self, keyword_1, keyword_2):
        '''Calculation Keyword Similarity'''
        keyword_1 = keyword_1.split(', ')
        keyword_2 = keyword_2.split(', ')
        embedding_keyword_1 = self.model.wv.get_mean_vector(keyword_1).reshape(1, -1)
        embedding_keyword_2 = self.model.wv.get_mean_vector(keyword_2).reshape(1, -1)
        
        keyword_sim = cosine_similarity(embedding_keyword_1, embedding_keyword_2)[0][0]
        
        return keyword_sim