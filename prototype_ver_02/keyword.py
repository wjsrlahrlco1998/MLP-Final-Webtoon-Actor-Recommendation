import pandas as pd
import numpy as np
import re
from konlpy.tag import Okt
from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity
import itertools
from tqdm.notebook import tqdm

class Keyword:
    
    def __init__(self, model_path):
        '''초기화'''
        self.model = FastText.load(model_path)
        self.okt = Okt()
    
    def preprocessing_sentence(self, sentence):
        '''문장 전처리'''
        sub_sentence = re.sub('[^가-힣 ]', '', sentence)
        
        return sub_sentence
    
    def preprocessing_keyword(self, keyword):
        '''키워드 전처리'''
        split_keyword = keyword.split(', ')
        
        return split_keyword
    
    def tokenize_sentence(self, sentence):
        '''문장 토큰화'''
        token_sentence = self.okt.morphs(sentence)
        
        return token_sentence
    
    def sentence_embedding(self, sentence):
        '''문장 임베딩'''
        sentence_vec = None
        
        for word in sentence:
            word_vec = self.model.wv.get_vector(word)
            
            if sentence_vec is None:
                sentence_vec = word_vec
            else:
                sentence_vec = sentence_vec + word_vec
        
        if sentence_vec is not None:
            sentence_vec = sentence_vec / len(sentence)
        
        return sentence_vec
    
    def max_sum_sim(self, embedding_content, embedding_candidate, adjective, top_n=5, nr_candidates=10):
        '''Max Sum Sim Algorithm'''
        distances_content = cosine_similarity(embedding_content, embedding_candidate)
        distances_candidate = cosine_similarity(embedding_candidate, embedding_candidate)
        
        words_idx = list(distances_content.argsort()[0][-nr_candidates:])
        words_vals = [adjective[index] for index in words_idx]
        distances_candidate = distances_candidate[np.ix_(words_idx, words_idx)]
        
        min_sim = np.inf
        candidate = None
        for combination in itertools.combinations(range(len(words_idx)), top_n):
            sim = sum([distances_candidate[i][j] for i in combination for j in combination if i != j])#자기 자신과의 유사도 제외.
            if sim < min_sim:
                candidate = combination
                min_sim = sim

        return [words_vals[idx] for idx in candidate]
        
    
    def extract_keyword(self, embedding_content, embedding_candidate, adjective, top_n=5, nr_candidates=10):
        '''키워드 추출'''
        keyword = self.max_sum_sim(embedding_content, embedding_candidate, adjective, top_n, nr_candidates)
        
        return keyword
        
    
    def run(self, df, col_content='content', col_keywords='keywords'):
        '''전체 실행'''
        sentence_list = df[col_content].to_list()
        adjective_list = [self.preprocessing_keyword(adjective) for adjective in tqdm(df[col_keywords])]
        
        # content 전처리
        clean_content = [self.preprocessing_sentence(sentence) for sentence in tqdm(sentence_list)]
        token_content = [self.tokenize_sentence(sentence) for sentence in tqdm(clean_content)]
        
        # content 임베딩
        embedding_contents = [self.sentence_embedding(token_sentence) for token_sentence in tqdm(token_content)]
        embedding_candidates = []
        for adjectives in tqdm(adjective_list):
            embedding_candidate = []
            for adjective in adjectives:
                embedding_candidate.append(self.model.wv.get_vector(adjective))
            embedding_candidates.append(embedding_candidate)
            
        # 키워드 추출
        keywords = [self.extract_keyword(embedding_contents[i].reshape(1, -1), embedding_candidates[i], adjective_list[i]) for i in tqdm(range(len(embedding_contents)))]
        
        return keywords
    
if __name__ == '__main__':
    # 모델 로드
    keyword = Keyword()
    
    # 데이터 로드
    # 조건 : keywords 10개 이상
    df = pd.read_csv('../text_data/Actors_keywords_clean.csv')

    keywords = keyword.run(df, col_content='content', col_keywords='keywords')
    
    print(keywords)