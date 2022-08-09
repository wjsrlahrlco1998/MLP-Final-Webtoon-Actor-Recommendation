# Package Load
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import PIL
import plotly.express as px
from PIL import Image
from keras.preprocessing import image as keras_image

# AI Model Module Load
from image_modules import AutoEncoder_2D, image_visualization, convert_image_to_anime
from text_modules import text_similarity

# Caching AI Model And Data
@st.cache(allow_output_mutation=True)
def load_actor_df():
    actor_df = pd.read_csv('./text_data/Actor_keywords.csv', encoding='utf-8-sig')
    return actor_df

@st.cache(allow_output_mutation=True)
def load_webtoon_df():
    webtoon_df = pd.read_csv('./text_data/Webtoon_keywords.csv', encoding='utf-8-sig')
    return webtoon_df

@st.cache(allow_output_mutation=True)
def load_keyword_score_df():
    keyword_score_df = pd.read_excel('./text_data/Keyword_score_table_3.xlsx')
    return keyword_score_df

@st.cache(allow_output_mutation=True)
def load_model_AE():
    model_AE = AutoEncoder_2D.ModelAE(model_path='./model_AutoEncoder/2D_AutoEncoder.ckpt')
    return model_AE

@st.cache(allow_output_mutation=True)
def load_model_orb():
    model_orb = image_visualization.ORB()
    return model_orb

@st.cache(allow_output_mutation=True)
def load_model_animeGan():
    model_animeGan = convert_image_to_anime.AnimeGAN()
    return model_animeGan

@st.cache(allow_output_mutation=True)
def load_model_keyword():
    model_keyword = text_similarity.Text()
    return model_keyword

# Setting Session
if 'session' not in st.session_state:
    st.session_state['session'] = 0

if 'session_1' not in st.session_state:
    st.session_state['session_1'] = 0
    
if 'character_list' not in st.session_state:
    st.session_state['character_list'] = []
    
if 'actor_list' not in st.session_state:
    st.session_state['actor_list'] = []

if 'recommend_n' not in st.session_state:
    st.session_state['recommend_n'] = 0

if 'upload_image' not in st.session_state:
    st.session_state['upload_image'] = None

if 'anime_image' not in st.session_state:
    st.session_state['anime_image'] = None
    
if 'character' not in st.session_state:
    st.session_state['character'] = [None] * 5

if 'user' not in st.session_state:
    st.session_state['user'] = [None] * 5

if 'user_1' not in st.session_state:
    st.session_state['user_1'] = [None] * 5

if 'actor_df' not in st.session_state:
    st.session_state['actor_df'] = load_actor_df()

if 'webtoon_df' not in st.session_state:
    st.session_state['webtoon_df'] = load_webtoon_df()

if 'keyword_score_df' not in st.session_state:
    st.session_state['keyword_score_df'] = load_keyword_score_df()
    
if 'model_keyword' not in st.session_state:
    st.session_state['model_keyword'] = load_model_keyword()

if 'model_rs50' not in st.session_state:
    st.session_state['model_AE'] = load_model_AE()
    
if 'model_animeGan' not in st.session_state:
    st.session_state['model_animeGan'] = load_model_animeGan()

if 'model_orb' not in st.session_state:
    st.session_state['model_orb'] = load_model_orb()

if 'init_state' not in st.session_state:
    st.session_state['init_state'] = False

if 'Feedback_id' not in st.session_state:
    st.session_state['Feedback_id'] = 0

if '5_user_image' not in st.session_state:
    st.session_state['5_user_image'] = [None] * 2

if 'character_image_1' not in st.session_state:
    st.session_state['character_image_1'] = None

if "result_option_3" not in st.session_state:
    st.session_state['result_option_3'] = [None] * 8

if "select_webtoon_3" not in st.session_state:
    st.session_state['select_webtoon_3'] = False

# Home image
if 'title_image_1' not in st.session_state:
    st.session_state['title_image_1'] = Image.open("./Home_image/Home_naver_webtoon.jpg")
    
# Home page
if st.session_state['init_state']:
    st.session_state['session'] = 0
    st.session_state['character_list'] = []
    st.session_state['actor_list'] = []
    st.session_state['recommend_n'] = 0
    st.session_state['upload_image'] = None
    st.session_state['anime_image'] = None
    st.session_state['character'] = [None] * 5
    st.session_state['user'] = [None] * 5
    st.session_state['init_state'] = False

st.markdown("## 📚AI 웹툰 등장인물 추천 시스템")
st.markdown("*당신의 사진을 웹툰으로 변환하고 웹툰 등장인물과 비교합니다.*")
st.markdown("*당신과 닮은 웹툰 등장인물을 찾아보세요~!*")

with st.expander("시스템 소개"):
    st.markdown(""" 웹툰 등장인물 추천 시스템은 웹툰 등장인물과 유사한 배우를 추천하거나 사용자와 웹툰의 등장인물 중 가장 비슷한 등장인물을 추천해주는 시스템 입니다. 이 시스템은 키워드 유사도, 이미지 유사도 등 다양한 지표를 활용하여 최종 유사도를 선정합니다.
    
**지금바로 웹툰 등장인물과 비교해보세요!!**

\* Code : [GitHub](https://github.com/wjsrlahrlco1998/MLP-Final-Webtoon-Actor-Recommendation)""")
    
with st.expander("기능 소개"):
    st.markdown("""1. 배우 추천 기능

 웹툰과 그 등장인물을 선택하여 어떤 배우가 가장 잘어울리는지 추천해주는 기능입니다.
 
2. 웹툰 등장인물 닮은꼴 찾기 기능

 내가 어떤 웹툰 등장인물과 가장 닮은지 순위별로 나타내주는 기능입니다.
 
3. 내가 원하는 웹툰 등장인물과 비교하기 기능

 내가 원하는 웹툰 등장인물의 사진과 나의 사진을 비교하여 얼마나 비슷한지 나타내주는 기능입니다.
 
4. 내가 원하는 웹툰 등장인물 사진을 직접 올려서 비교하기 기능

 내가 원하는 웹툰 등장인물의 사진과 나의 사진을 모두 내가 직접올려서 비교할 수 있는 기능입니다.
 
5. 웹툰 캐릭터 변환 기능

나의 사진을 웹툰 그림체로 변환하는 기능입니다.""")    

with st.expander("프로젝트 인원 소개"):
    st.markdown("""**1. 팀원 구성 및 역할**
- 김지원(팀장) - PM
    - Project Management
    - Data collection
    - Data preprocessing
- 김동현(팀원) - PE
    - Data collection
    - Data preprocessing
    - StyleGAN2 Model Training
- 구광모(팀원) - PE
    - Data collection
    - Data preprocessing
    - FastText Model Training
- 박지성(팀원) - PL
    - Data collection
    - Data preprocessing
    - AutoEncoder Model Training
    - Model & Function Modulization
    - Apply AI Model
    - Web Front-End & Back-End
- 정경희(팀원) - PE
    - Data collection
    - Data preprocessing
    - FastText Model Training
- 윤형섭(팀원) - PE
    - Data collection
    - Web Front-End & Back-End
""")

st.image(st.session_state['title_image_1'], use_column_width='always')