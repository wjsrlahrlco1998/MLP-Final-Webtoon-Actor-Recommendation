import pandas as pd
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing import image as keras_image
from PIL import Image
import warnings
import numpy as np

class User:
    
    def __init__(self, model_gan, model_RS50, model_keyword, model_ORB):
        self.model_gan = model_gan
        self.model_RS50 = model_RS50
        self.model_keyword = model_keyword
        self.model_ORB = model_ORB
        
        self.star_list = ['마동석', '이정재', '공유']
        self.drama_star = ['박은빈', '주헌영']
    
    def upload_character_image(self):
        state = True
        
        while state:
            print(f"\n{'=' * 15} 웹툰 등장인물 사진 업로드 {'=' * 15}")
            print(f"""* 사진은 jpg 형식으로 올려주세요.""")
            print('=' * 53)
            upload_image_path = input('>>> 입력(현재는 경로로 대체) : ')
            try:
                upload_image = keras_image.load_img(f'{upload_image_path}.jpg', target_size=(224, 224))
                plt.figure(figsize=(8, 5))
                plt.imshow(upload_image)
                plt.axis('off')
                plt.show()
                state = False
            except:
                print("Error : 잘못된 경로 혹은 이미지입니다.")
        
        return upload_image
    
    def input_character_info(self):
        state = True
        
        while state:
            print(f"\n{'=' * 17} 등장인물 정보 입력 {'=' * 17}")
            print(f'''* 키워드는 반드시 하나이상 입력해야한다.
* 키워드를 여러개 입력할 때, 구분자는 ", "이다.
* 예시) 바보같다, 사랑스럽다, 둔하다''')
            print(('=' * 51) +"\n")
            character_name = input('>>> 이름 입력 : ')
            character_keyword = input('>>> 키워드 입력(구분자는 ",") : ')
            
            
            if len(character_keyword.strip()) > 0:
                state = False
            else: 
                print("Error : 키워드는 반드시 하나이상 입력해야합니다")
                
        return character_name, character_keyword
    
    def upload_actor_image(self):
        state = True
        
        while state:
            print(f"\n{'=' * 15} 배우 사진 업로드 {'=' * 15}")
            print(f"""* 사진은 jpg 형식으로 올려주세요.""")
            print('=' * 48)
            upload_image_path = input('>>> 입력(현재는 경로로 대체) : ')
            try:
                upload_image = keras_image.load_img(f'{upload_image_path}.jpg', target_size=(224, 224))
                plt.figure(figsize=(8, 5))
                plt.imshow(upload_image)
                plt.axis('off')
                plt.show()
                state = False
            except:
                print("Error : 잘못된 경로 혹은 이미지입니다.")
        
        return upload_image
    
    def input_actor_info(self):
        state = True
        
        print(f"\n{'=' * 17} 배우 정보 입력 {'=' * 17}")
        print(f'''* 키워드는 반드시 하나이상 입력해야한다.
* 키워드를 여러개 입력할 때, 구분자는 ", "이다.
* 예시) 바보같다, 사랑스럽다, 둔하다''')
        print(('=' * 49) +"\n")
        actor_name = input('>>> 이름 입력 : ')
        actor_keyword = input('>>> 키워드 입력(구분자는 ",") : ')

        if len(actor_keyword.strip()) > 0:
            state = False
        else: 
            print("Error : 키워드는 반드시 하나이상 입력해야합니다")
        
        if actor_name in self.star_list:
            print(f"{actor_name} 배우는 한류스타입니다.\n가중치가 부여됩니다.")
        else:
            print(f"{actor_name} 배우는 한류스타가 아닙니다.\n가중치가 부여되지 않습니다.")
        if actor_name in self.drama_star:
            print(f"{actor_name} 배우는 드라마 스타입니다.\n가중치가 부여됩니다.")
        else:
            print(f"{actor_name} 배우는 드라마 스타가 아닙니다.\n가중치가 부여되지 않습니다.")
        
        return actor_name, actor_keyword
    
    def convert_real_to_anime(self, actor_image):
        conv_anime_image = self.model_gan.run(actor_image, 'test', path='./')
        
        return conv_anime_image
    
    def compare_image(self, webtoon_image, actor_image):
        image_similarity = self.model_RS50.run(webtoon_image, actor_image)
        
        return round(image_similarity[0][0] * 100, 2)
    
    def compare_keyword(self, webtoon_keywords, actor_keywords):
        keyword_similarity = self.model_keyword.keyword_similarity(webtoon_keywords, actor_keywords)
        
        return round(keyword_similarity * 100, 2)
    
    def compare_visualization(self, actor_image, anime_actor_image, charactor_image):
        actor_image = self.model_RS50.resize(actor_image, 256)
        actor_image = self.model_RS50.convert_type(actor_image, TYPE=1)
        actor_image = self.model_ORB.img_to_rgb(actor_image)
        
        anime_actor_image = self.model_RS50.convert_type(anime_actor_image, TYPE=1)
        charactor_image = self.model_RS50.convert_type(charactor_image, TYPE=1)
        
        knn_image = self.model_ORB.run(anime_actor_image, charactor_image)
        
        show_image = np.concatenate([actor_image, knn_image], axis=1)
        
        plt.imshow(show_image)
        plt.axis('off')
        plt.show()
    
    def result(self, actor_name, actor_keyword, charactor_name, 
               charactor_keyword, actor_image, anime_actor_image, character_image, keyword_similarity, image_similarity):
        total_score = round(keyword_similarity * 0.2, 2) + round(image_similarity * 0.8, 2)
        if actor_name in self.star_list:
            total_score = total_score * 1.1
            total_score = round(total_score, 2)
        
        if actor_name in self.drama_star:
            total_score = total_score * 1.1
            total_score = round(total_score, 2)
        
        print(f"{'=' * 19} << 결과 >> {'=' * 19}")
        self.compare_visualization(actor_image, anime_actor_image, character_image)
        print(f"배우 이름: {actor_name}")
        print(f"배우 키워드 : {actor_keyword}")
        print(f"등장인물 이름 : {charactor_name}")
        print(f"등장인물 키워드 : {charactor_keyword}\n")
        print(f"키워드 유사도 : {keyword_similarity}")
        print(f"이미지 유사도 : {image_similarity}")
        print(f"전체 점수 : {total_score}")
        print(f"{'=' * 19}==========={'=' * 19}")
        
    
    def run(self):
        
        # 1. 사용자의 웹툰 등장인물 사진 업로드
        character_image = self.upload_character_image()
        
        # 2. 사용자의 웹툰 등장인물 키워드 입력
        charactor_name, character_keywords = self.input_character_info()
        
        # 3. 사용자의 배우 사진 업로드
        actor_image = self.upload_actor_image()
        
        # 4. 사용자의 배우 키워드 입력
        actor_name, actor_keywords = self.input_actor_info()
        
        # 5. 배우 사진 웹툰화
        anime_actor_image = self.convert_real_to_anime(actor_image)
        
        # 6. 사진 비교
        image_similarity = self.compare_image(character_image, anime_actor_image)
        
        # 7. 키워드 비교
        keyword_similarity = self.compare_keyword(character_keywords, actor_keywords)
        
        # 8. 최종 결과
        self.result(actor_name, actor_keywords, charactor_name, 
               character_keywords, actor_image, anime_actor_image, character_image, keyword_similarity, image_similarity)