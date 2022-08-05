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
from image_modules import image_similarity, image_visualization, convert_image_to_anime
from text_modules import text_similarity

# Load AI Model from session
model_keyword = st.session_state['model_keyword']
model_rs50 = st.session_state['model_rs50']
model_animeGan = st.session_state['model_animeGan']
model_orb = st.session_state['model_orb']

# Define radar_chart
def radar_chart(age_score, keyword_score, image_score, sex_score, harmony):
    df = pd.DataFrame(dict(
    r=[age_score,
       keyword_score,
       image_score,
       sex_score,
       harmony],
    theta=['나이','키워드 유사도','이미지 유사도',
           '성별', '조화']))
    fig = px.line_polar(df, r='r', theta='theta', line_close=True, range_r =(0, 100))
    st.write(fig)
    
st.title("이 등장인물은 나와 닮았을까?")

# Page
with st.container():
    col_1, col_2 = st.columns(2)
    
    with col_1:
        with st.form("form_1", clear_on_submit=False):
            character_image_bytes = st.file_uploader("1. 웹툰 등장인물 사진을 업로드하세요.", type=['jpg', 'png'], accept_multiple_files=False)
            character_name = st.text_input("2. 등장인물의 이름을 입력하세요.", placeholder="여기에 입력하세요.")
            character_age = st.number_input("3. 나이를 입력하세요.", min_value=0, value=0)
            character_sex = st.radio("4. 성별을 고르세요.", ('남', '여'), horizontal=True)
            character_keyword = st.text_input("5. 등장인물의 키워드를 입력하세요.", placeholder="구분자는 ', '입니다.")
            submitted_1 = st.form_submit_button("입력완료")
        if submitted_1:
            st.session_state['character'][0] = character_image_bytes
            st.session_state['character'][1] = character_name
            st.session_state['character'][2] = character_age
            st.session_state['character'][3] = character_sex
            st.session_state['character'][4] = character_keyword
    with col_2:
        with st.form("form_2", clear_on_submit=False):
            user_image_bytes = st.file_uploader("1. 등장인물과 비교할 사진을 업로드하세요.", type=['jpg', 'png'], accept_multiple_files=False)
            user_name = st.text_input("2. 사진의 이름을 입력하세요.", placeholder="여기에 입력하세요.")
            user_age = st.number_input("3. 나이를 입력하세요.", min_value=0, max_value=120, value=0)
            user_sex = st.radio("4. 성별을 고르세요.", ('남', '여'), horizontal=True)
            user_keyword = st.text_input("5. 사진의 키워드를 입력하세요.", placeholder="구분자는 ', '입니다.")
            submitted_2 = st.form_submit_button("입력완료")
        if submitted_2:
            st.session_state['user'][0] = user_image_bytes
            st.session_state['user'][1] = user_name
            st.session_state['user'][2] = user_age
            st.session_state['user'][3] = user_sex
            st.session_state['user'][4] = user_keyword

with st.container():
    info_col_1, info_col_2, info_col_3, info_col_4 = st.columns(4)
    if st.session_state['character'][0]:
        with info_col_1:
            st.image(st.session_state['character'][0])
        with info_col_2:
            st.markdown(f"- 이름   : {st.session_state['character'][1]}")
            st.markdown(f"- 나이   : {st.session_state['character'][2]}")
            st.markdown(f"- 성별   : {st.session_state['character'][3]}")
            st.markdown(f"- 키워드 : {st.session_state['character'][4]}")
    if st.session_state['user'][0]:
        with info_col_3:
            st.image(st.session_state['user'][0])
        with info_col_4:
            st.markdown(f"- 이름   : {st.session_state['user'][1]}")
            st.markdown(f"- 나이   : {st.session_state['user'][2]}")
            st.markdown(f"- 성별   : {st.session_state['user'][3]}")
            st.markdown(f"- 키워드 : {st.session_state['user'][4]}")
with st.container():
    _, _, col_middle, _, _ = st.columns(5)
    
    with col_middle:
        compare_button = st.button("비교시작", key='compare')

try:
    if compare_button:
        # 이미지 변환
        character_image = Image.open(character_image_bytes)
        user_image = Image.open(user_image_bytes)
        user_anime_image = model_animeGan.run(user_image.convert("RGB"))

        # 점수 계산
        if user_age > character_age:
            age_score = character_age / user_age
        else:
            age_score = user_age / character_age 
        image_score = model_rs50.run(user_anime_image, character_image)
        keyword_score = model_keyword.get_keyword_similarity(character_keyword, user_keyword)
        if character_sex == user_sex:
            sex_score = 1
        else:
            sex_score = 0.5
        total_score = (0.2 * keyword_score) + (0.6 * image_score) + (0.1 * age_score) + (0.1 * sex_score)

        # orb_knn 이미지
        cv2_character_image = cv2.cvtColor(np.array(character_image), cv2.COLOR_RGB2BGR)
        cv2_user_image = cv2.cvtColor(np.array(user_image), cv2.COLOR_RGB2BGR)
        cv2_user_image = cv2.resize(cv2_user_image, (256, 256))
        cv2_user_image = cv2.cvtColor(cv2_user_image, cv2.COLOR_BGR2RGB)
        cv2_user_anime_image = model_rs50.convert_type(user_anime_image, TYPE=1, GRAY=False)

        orb_knn_image = model_orb.run(cv2_user_anime_image, cv2_character_image)
        concatenate_image = np.concatenate([cv2_user_image, orb_knn_image], axis=1)
        st.image(concatenate_image, use_column_width='always', caption=f"\"{character_name}와(과) {user_name} 비교 결과")

        radar_chart(round(age_score * 100, 2), round(keyword_score * 100, 2), round(image_score * 100, 2), sex_score * 100, round(total_score * 100, 2))
except:
    st.error("얼굴을 인식하지 못했습니다. 다른 사용자 사진을 업로드 해주세요.")