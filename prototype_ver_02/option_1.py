import pandas as pd
import PIL
import cv2
from PIL import Image
from keras.preprocessing import image
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numpy as np

class User:
    
    def __init__(self, webtoon_df, actor_df, keyword_score_df, model_ORB, model_RN, image_score_rate=0.7):
        '''초기화'''
        self.webtoon_df = webtoon_df
        self.actor_df = actor_df
        self.keyword_score_df = keyword_score_df
        # self.image_score_df = image_score_df
        self.model_ORB = model_ORB
        self.model_RN = model_RN
        self.image_score_rate = image_score_rate
        self.keyword_score_rate = 1 - image_score_rate
    
    def select_webtoon_title(self, title_list):
        '''웹툰 제목 선택'''
        print(f"\n{'='*25} 웹툰 제목 선택 {'='*25}")
        for i in range(len(title_list)):
            title = f'{i+1}. {title_list[i]}'
            if (i+1) % 3 == 0:
                print("{0:<20s}".format(title))
            else:
                print("{0:<20s}".format(title), end='')
        print(f"\n{'='*68}\n")
        select_num = int(input(">>> 번호를 입력하세요 : "))
        select_title = title_list[select_num - 1]
        
        return select_title
    
    def select_webtoon_charater(self, character_list, select_title):
        '''웹툰 등장인물 선택'''
        fig_row = len(character_list) // 3 + 1

        print(f"\n{'=' * 26} 등장 인물 선택 {'=' * 26}")
        plt.figure(figsize=(10, 5))
        for i in range(len(character_list)):
            try:
                print(f'{i + 1}. {character_list[i]}')
                charactor_image = cv2.imread(f'./image_data/Webtoon/{select_title}_{character_list[i]}.jpg', flags=cv2.IMREAD_COLOR)
                charactor_image = self.model_ORB.img_resize(charactor_image)
                charactor_image = self.model_ORB.img_to_rgb(charactor_image)
                plt.subplot(fig_row, 3, i + 1)
                plt.imshow(charactor_image)
                plt.title(f"{i+1}")
                plt.axis('off')
            except:
                # print(f'{i + 1}. {character_list[i]} 이미지 없음')
                pass
        plt.show()
        print(f"\n{'='*70}\n")
        select_num = int(input('>>> 번호를 입력하세요 : '))
        select_character = character_list[select_num - 1]

        return select_character
    
    def scopping_age_range(self):
        '''나이를 지정하여 추천범위 지정'''
        print(f"\n{'=' * 14} 나이 범위 지정 {'=' * 14}")
        print(f""" * 숫자로만 입력하세요.""")
        print(('=' * 44)+"\n")
        min_age = int(input(">>> 최소 나이 입력: "))
        max_age = int(input(">>> 최대 나이 입력: "))
        
        return min_age, max_age
    
    def filter_actor(self, min_age, max_age, character_sex):
        '''배우 필터링'''
        filter_actor_df = self.actor_df[(self.actor_df['Sex'] == character_sex) &
                                       (self.actor_df['age'] >= min_age) &
                                       (self.actor_df['age'] <= max_age)]
        
        return filter_actor_df
    
    def select_actor_n(self, max_actor_num):
        '''추천받을 배우의 수 지정'''
        print(f"\n{'=' * 13} 추천받을 배우의 수 지정 {'=' * 13}")
        print(f"""* 필터링된 배우의 최대 수 : {max_actor_num}""")
        print(('=' * 51)+"\n")
        actor_n = int(input(">>> 추천받을 배우의 수 : "))
        
        return actor_n
    
    def top_n_actor(self, actor_n, select_title, select_character, filter_actor_df):
        '''top_n 명의 actor 추천'''
        filter_actor_df = filter_actor_df.reset_index(drop=True)
        
        keyword_score_dict = dict()
        image_score_dict = dict()
        total_score_dict = dict()
        
        webtoon_image = image.load_img(f'./image_data/Webtoon/{select_title}_{select_character}.jpg', target_size=(224, 224))
        
        # 키워드 점수 테이블에서 점수 구하기
        for i in tqdm(range(len(filter_actor_df))):
            actor_name = filter_actor_df['Name'][i]
            actor_age = filter_actor_df['age'][i]
            actor_sex = filter_actor_df['Sex'][i]
            try:
                actor_image = image.load_img(f'./image_data/Actor2webtoon/webtoon_{actor_name}_{actor_age}_{actor_sex}.jpg', target_size=(224, 224))
            except:
                actor_image = image.load_img(f'./image_data/Actor/{actor_name}_{actor_age}_{actor_sex}.jpg', target_size=(224, 224))
            
            keyword_score = self.keyword_score_df[(self.keyword_score_df['Title']==select_title) & 
                                 (self.keyword_score_df['Name']==select_character)][f'{actor_name}_{actor_age}_{actor_sex}'].values[0]
            image_score = self.model_RN.run(webtoon_image, actor_image)
            total_score = (keyword_score * self.keyword_score_rate) + (image_score * self.image_score_rate)
            
            keyword_score_dict[f'{actor_name}_{actor_age}_{actor_sex}'] = keyword_score
            image_score_dict[f'{actor_name}_{actor_age}_{actor_sex}'] = image_score
            total_score_dict[f'{actor_name}_{actor_age}_{actor_sex}'] = total_score
        
        total_score_dict = sorted(total_score_dict.items(), key = lambda x : x[1], reverse = True)[:actor_n]
        keyword_score_dict = sorted(keyword_score_dict.items(), key = lambda x : x[1], reverse = True)[:actor_n]
        image_score_dict = sorted(image_score_dict.items(), key = lambda x : x[1], reverse = True)
        
        return keyword_score_dict, image_score_dict, total_score_dict
    
    def option_1_result(self, select_title, select_character, total_score_dict, keyword_score_dict, image_score_dict):
        '''결과 보여주기'''
        webtoon_image = cv2.imread(f'./image_data/Webtoon/{select_title}_{select_character}.jpg', flags=cv2.IMREAD_COLOR)
        
        webtoon_keyword = self.webtoon_df[(self.webtoon_df['Title'] == select_title) &
                                         (self.webtoon_df['Name'] == select_character)]['5_keywords'].values[0]
        
        # 2. 결과
        print(f"{'='*19} << 결과 >> {'='*19}")
        print(f"* 점수 산정 : 키워드 {round(self.keyword_score_rate * 100, 0)}%, 이미지 {round(self.image_score_rate * 100, 2)}%")
        count = 0
        for name, _ in tqdm(total_score_dict):
            total_score = total_score_dict[count][1]
            keyword_score = self.keyword_score_df[(self.keyword_score_df['Title']==select_title) &
                                                 (self.keyword_score_df['Name']==select_character)][name].values[0]
            for img_name, score in image_score_dict:
                if img_name == name:
                    image_score = score
            
            try:
                _ = image.load_img(f'./image_data/Actor2Webtoon/webtoon_{name}.jpg', target_size=(224, 224))
                actor_image = cv2.imread(f'./image_data/Actor2Webtoon/webtoon_{name}.jpg', flags=cv2.IMREAD_COLOR)
            except:
                actor_image = cv2.imread(f'./image_data/Actor/{name}.jpg', flags=cv2.IMREAD_COLOR)
                
            actor_info = name.split('_')
            actor_name = actor_info[0]
            actor_age = int(actor_info[1])
            actor_sex = actor_info[2]
            print('=' * 51)
            print(f"배우 이름 : {actor_name}\n배우 나이 : {actor_age}\n배우 성별 : {actor_sex}")
            
            actor_keyword = self.actor_df[(self.actor_df['Name'] == actor_name) &
                                         (self.actor_df['age'] == actor_age) &
                                         (self.actor_df['Sex'] == actor_sex)]['5_keywords'].values[0]
            
            
            # 1. ORB
            real_actor_image = cv2.imread(f'./image_data/Actor/{name}.jpg')
            real_actor_image = self.model_ORB.img_resize(real_actor_image)
            real_actor_image = self.model_ORB.img_to_rgb(real_actor_image)
            orb_knn_image = self.model_ORB.run(actor_image, webtoon_image, show=False)
            concatenate_image = np.concatenate([real_actor_image, orb_knn_image], axis=1)
            plt.imshow(concatenate_image)
            plt.axis('off')
            plt.show()
            
            print(f"Character Keywords : {webtoon_keyword}")
            print(f"Actor keywords : {actor_keyword}\n")
            print(f"Keyword Similarity     : {round(keyword_score * 100, 2)}%")
            print(f"Image Similarity       : {round(image_score[0][0] * 100, 2)}%")
            print(f"** Total Similarity ** : {round(total_score[0][0] * 100, 2)}%")
            print('=' * 51)
            count += 1
            
            
    def recommend_actor(self):
        '''배우 추천'''
        select_title = self.select_webtoon_title(self.webtoon_df['Title'].unique())
        
        character_list = self.webtoon_df[self.webtoon_df['Title'] == select_title]['Name'].to_list()
        select_character = self.select_webtoon_charater(select_title=select_title, character_list=character_list)
        character_sex = self.webtoon_df[(self.webtoon_df['Title'] == select_title) & 
                        (self.webtoon_df['Name'] == select_character)]['Sex'].values[0]
        
        min_age, max_age = self.scopping_age_range()
        filter_actor_df = self.filter_actor(min_age, max_age, character_sex)
        
        select_actor_n = self.select_actor_n(len(filter_actor_df))
        
        keyword_score_dict, image_score_dict, total_score_dict = self.top_n_actor(select_actor_n, 
                                                                                  select_title, select_character, filter_actor_df)
        
        self.option_1_result(select_title, select_character, total_score_dict, keyword_score_dict, image_score_dict)