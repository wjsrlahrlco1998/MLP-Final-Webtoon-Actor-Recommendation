from model_rs50 import RN50
import pandas as pd

class User:
    
    def __init__(self, webtoon_df, actor_df, model_obj):
        self.webtoon_df = webtoon_df
        self.actor_df = actor_df

        self.model = model_obj
    
    # 웹툰 목록을 사용자에게 보여줌 -> 사용자 선택 반환
    def select_webtoon(self):
        webtoon_titles = self.webtoon_df['웹툰이름 '].unique()

        for i in range(len(webtoon_titles)):
            print(f"{i + 1}. {webtoon_titles[i]}")

        select_num = int(input(f'웹툰 번호 선택 : '))
        select_title = webtoon_titles[select_num - 1]
        
        return select_title

    # 웹툰 등장인물 목록을 사용자에게 보여줌 -> 사용자 선택 반환
    def select_webtoon_character(self, webtoon_title):
        webtoon_characters = self.webtoon_df[self.webtoon_df['웹툰이름 '] == webtoon_title]['등장인물'].to_list()

        for i in range(len(webtoon_characters)):
            print(f"{i + 1}. {webtoon_characters[i]}")

        select_num = int(input(f'등장인물 번호 선택 : '))
        select_character = webtoon_characters[select_num - 1]
        
        return select_character

    # 등장인물과 배우와의 유사도를 비교 후 dict형 반환
    def compute_sim(self, title, character):
        score_dict = dict()

        webtoon_character_sex = self.webtoon_df.loc[(self.webtoon_df['웹툰이름 '] == title) & (self.webtoon_df['등장인물'] == character),
                                           '성별'].values[0]
        webtoon_character_age = self.webtoon_df.loc[(self.webtoon_df['웹툰이름 '] == title) & (self.webtoon_df['등장인물'] == character),
                                           '나이'].values[0]
        webtoon_character_age = int(re.sub('[^0-9]', '', webtoon_character_age))
        webtoon_image = cv2.imread(f'../image_data/Webtoon/{title}_{character}.jpg', cv2.IMREAD_COLOR)

        if webtoon_character_sex in ['남', '여']:
            for actor_name in self.actor_df[self.actor_df['Sex'] == webtoon_character_sex]['Name'].to_list():
                actor_age = int(self.actor_df[self.actor_df['Name'] == actor_name]['Age'].values[0])
                if (webtoon_character_age - 5 <= actor_age) and (webtoon_character_age + 25 >= actor_age):
                    actor_image = cv2.imread(f'../image_data/Actor2Webtoon/webtoon_{actor_name}.jpg', cv2.IMREAD_COLOR)
                    score = self.model.run(webtoon_image, actor_image)
                    score_dict[actor_name] = score
        else:
            for actor_name in self.actor_df['Name'].to_list():
                actor_age = int(self.actor_df[self.actor_df['Name'] == actor_name]['Age'].values[0])
                if (webtoon_character_age < actor_age) and (webtoon_character_age + 20 > actor_age):
                    actor_image = cv2.imread(f'../image_data/Actor2Webtoon/webtoon_{actor_name}.jpg', cv2.IMREAD_COLOR)
                    score = self.model.run(webtoon_image, actor_image)
                    score_dict[actor_name] = score

        return score_dict

    # 유사도 비교 결과를 사용자에게 보여줌
    def show_result(self, score_dict, title, character):
        webtoon_image = cv2.imread(f'../image_data/Webtoon/{title}_{character}.jpg', cv2.IMREAD_COLOR)
        
        score_dict = sorted(score_dict.items(), key = lambda x : x[1], reverse=True)

        for name, score in score_dict[:5]:
            print(f'이름 : {name}\n점수 : {score}')
            actor_image = cv2.imread(f'../image_data/Actor/{name}.jpg', cv2.IMREAD_COLOR)
            actor_image = cv2.cvtColor(actor_image, cv2.COLOR_BGR2RGB)
            plt.imshow(actor_image)
            plt.axis('off')
            plt.show()
            
    # run
    def run(self):
        title = self.select_webtoon()
        character = self.select_webtoon_character(title)
        score_dict = self.compute_sim(title, character)
        self.show_result(score_dict, title, character)

if __name__ == '__main__':
    actor_df = pd.read_csv('../text_data/Actor.csv', encoding = 'utf-8-sig')
    webtoon_df = pd.read_excel('../text_data/Webtoon.xlsx')
    
    model = RS50()
    user = User(webtoon_df, actor_df, rn_model)
    user.run()