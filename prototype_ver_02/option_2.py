import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import warnings
import numpy as np

class User:
    
    def __init__(self, webtoon_df, model_ORB, model_key, model_RS, model_gan, image_score_rate=0.7):
        '''초기화'''
        self.webtoon_df = webtoon_df
        self.model_ORB = model_ORB
        self.model_key = model_key
        self.model_RS = model_RS
        self.model_gan = model_gan
        self.image_score_rate = 0.7
        self.keyword_score_rate = 1 - self.image_score_rate
    
    def select_webtoon_title(self, title_list):
        '''웹툰 선택'''
        state = True
        while state:
            print(f"\n{'='*25} 웹툰 제목 선택 {'='*25}")

            for i in range(len(title_list)):
                text = f'{i + 1}. {title_list[i]}'
                if (i + 1) % 3 == 0:
                    print("{0:<20s}".format(text))
                else:
                    print("{0:<20s}".format(text), end='')

            print(f"\n{'='*68}\n")
            select_num = int(input('>>> 번호를 입력하세요 : '))
            try:
                select_title = title_list[select_num - 1]
                if select_num < 1:
                    raise("Error")
                state = False
            except:
                print("Error : 잘못된 번호입니다.")
        
        return select_title
    
    def select_webtoon_character(self, character_list, select_title):
        '''등장인물 선택'''
        fig_row = len(character_list) // 3 + 1
        
        state = True
        
        while state:
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
                    pass
            plt.show()
            print(f"\n{'='*70}\n")
            select_num = int(input('>>> 번호를 입력하세요 : '))
            try:
                select_character = character_list[select_num - 1]
                if select_num < 1:
                    raise("Error")
                state = False
            except:
                print("Error : 잘못된 번호입니다.")

        return select_character
    
    def get_character_image(self, title, name):
        '''등장인물 사진 가져오기'''
        select_character_image = cv2.imread(f'./image_data/Webtoon/{title}_{name}.jpg', flags=cv2.IMREAD_COLOR)
        select_character_image = self.model_ORB.img_resize(select_character_image)
        select_character_image = self.model_ORB.img_to_rgb(select_character_image)

        return select_character_image
    
    def show_character_image(self, image):
        '''등장인물 사진 보여주기'''
        print(f"\n{'=' * 13} 선택된 등장인물 {'=' * 13}")
        plt.figure(figsize=(8, 5))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        print('=' * 43)
    
    def upload_user_image(self):
        '''사용자 사진 업로드'''
        state = True
        while state:
            print(f"\n{'=' * 15} 사진 업로드 {'=' * 15}")
            print(f"""* 사진은 jpg 형식으로 올려주세요.""")
            print('=' * 45)
            upload_image_path = input('>>> 입력(현재는 경로로 대체) : ')
            try:
                upload_image = Image.open(f'{upload_image_path}.jpg')
                conv_upload_image = upload_image.convert("RGB")
                plt.figure(figsize=(8, 5))
                plt.imshow(conv_upload_image)
                plt.axis('off')
                plt.show()
                state = False
            except:
                print("Error : 잘못된 경로 or 이미지입니다.")

            print('=' * 45)
    
        return upload_image, conv_upload_image
    
    def input_user_keyword(self):
        '''사용자 키워드 입력'''
        state = True
        while state:
            print(f"\n{'=' * 17} 키워드 입력 {'=' * 17}")
            print(f'''* 키워드는 반드시 하나이상 입력해야한다.
* 키워드를 여러개 입력할 때, 구분자는 ", "이다.
* 예시) 바보같다, 사랑스럽다, 둔하다''')
            print(('=' * 47) +"\n")
            user_keyword = input('>>> 입력(구분자는 ",") : ')
        
            if len(user_keyword.strip()) > 0:
                state = False
            else: 
                print("Error : 키워드는 반드시 하나이상 입력해야합니다")
        
        return user_keyword
    
    def option_2_result(self, user_image, orb_knn_image, select_character_keyword, user_keyword, keyword_score, 
                       image_score):
        '''option 2의 실행결과'''
        user_image = self.model_RS.convert_type(user_image, TYPE=1)
        user_image = self.model_ORB.img_resize(user_image)
        user_image = self.model_ORB.img_to_rgb(user_image)
        show_image = np.concatenate([user_image, orb_knn_image], axis=1)
        print()
        print(f"{'=' * 19} << 결과 >> {'=' * 19}")
        plt.imshow(show_image)
        plt.axis('off')
        plt.show()
        print(f"Character keywords : {select_character_keyword}")
        print(f"User keywords : {user_keyword}\n")
        print(f"Keyword Similarity : {keyword_score}%")
        print(f"Image Similarity : {image_score}%\n")
        print(f"Total Similarity : {round((keyword_score * self.keyword_score_rate) + (image_score * self.image_score_rate), 2)}%")
        print(f"{'=' * 19}==========={'=' * 19}")
    
    def run_option_2(self):
        '''Option2 실행'''
        
        # 1. 사용자 웹툰 선택
        title_list = self.webtoon_df['Title'].unique()
        select_title = self.select_webtoon_title(title_list)
        
        # 2. 사용자 등장인물 선택
        character_list = self.webtoon_df[self.webtoon_df['Title'] == select_title]['Name'].to_list()
        select_character = self.select_webtoon_character(character_list, select_title)
        
        # 3. 등장인물 데이터 가져오기
        select_character_image = self.get_character_image(select_title, select_character) # 이미지
        select_character_keyword = self.webtoon_df[(self.webtoon_df['Title'] == select_title) & 
                                                  (self.webtoon_df['Name'] == select_character)]['5_keywords'].values[0] # 키워드
        self.show_character_image(select_character_image) # 등장인물 print
        select_character_image = self.model_ORB.img_to_bgr(select_character_image) # 등장인물 사진 변환 (입력값)
        
        # 4. 사용자 사진 업로드 및 변환
        state = True
        while state:
            try:
                user_image, upload_image = self.upload_user_image()
                anime_user_image = self.model_gan.run(upload_image, name='User', path='./user/')
                state = False
            except:
                print("Error : 얼굴을 인식할 수 없는 사진입니다.")
        
        # 5. 사용자 키워드 입력
        user_keyword = self.input_user_keyword()
        
        # 6. 유사도 구하기
        keyword_score = round(self.model_key.keyword_similarity(select_character_keyword, user_keyword) * 100, 2) # 키워드 유사도
        image_score = round(self.model_RS.run(anime_user_image, select_character_image)[0][0] * 100, 2) # 이미지 유사도
        
        # 7. ORB Algorithm으로 극적 시각화
        anime_user_image = self.model_RS.convert_type(anime_user_image, TYPE=1, GRAY=False)
        orb_knn_image = self.model_ORB.run(anime_user_image, select_character_image, show=False)
        
        # 8. 결과 보여주기
        self.option_2_result(user_image, orb_knn_image, select_character_keyword, user_keyword, keyword_score, image_score)
    