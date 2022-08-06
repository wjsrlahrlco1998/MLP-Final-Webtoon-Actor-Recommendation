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

# Load AI Model and Data from session
model_keyword = st.session_state['model_keyword']
model_rs50 = st.session_state['model_rs50']
model_animeGan = st.session_state['model_animeGan']
model_orb = st.session_state['model_orb']
    
webtoon_df = st.session_state['webtoon_df']
webtoon_titles = webtoon_df['Title'].unique().tolist()

# Page
st.title("비교하고 싶은 등장인물을 선택하여 비교하세요!")
main_col_1, main_col_2 = st.columns(2)
with main_col_1:
    with st.form('form_1', clear_on_submit=False):
        selected_title = st.selectbox("1. 원하는 웹툰 제목을 검색하세요!!", set(webtoon_titles), key="sbox1", disabled=False)
        submitted_1 = st.form_submit_button('완료')

    with st.form('form_2', clear_on_submit=False):
        character_list = webtoon_df[webtoon_df['Title']==selected_title]['Name'].to_list()
        selected_character = st.radio("2. 원하는 등장인물을 선택하세요!!", set(character_list))
        submitted_2 = st.form_submit_button('완료')

with main_col_2:
    with st.form('form_3', clear_on_submit=False):
        user_image_bytes = st.file_uploader("1. 등장인물과 비교할 사진을 업로드하세요.", type=['jpg', 'png'], accept_multiple_files=False)
        user_name = st.text_input("2. 사진의 이름을 입력하세요.", placeholder="여기에 입력하세요.")
        user_age = st.number_input("3. 나이를 입력하세요", min_value=0, max_value=120, value=0)
        user_sex = st.radio("4. 성별을 고르세요.", ('남', '여'), horizontal=True)
        user_keyword = st.text_input("5. 사진의 키워드를 입력하세요.", placeholder="구분자는 ', '입니다.")
        submitted_3 = st.form_submit_button("입력완료")
        
        if submitted_3:
            st.session_state['user_1'][0] = user_image_bytes
            st.session_state['user_1'][1] = user_name
            st.session_state['user_1'][2] = user_age
            st.session_state['user_1'][3] = user_sex
            st.session_state['user_1'][4] = user_keyword
        
with st.container():        
    if submitted_2 or selected_character:
        character_image = Image.open(f"./image_data/Webtoon/{selected_title}_{selected_character}.jpg")
        character_age = webtoon_df[(webtoon_df['Title'] == selected_title) & (webtoon_df['Name']==selected_character)]['Age'].values[0]
        character_sex = webtoon_df[(webtoon_df['Title'] == selected_title) & (webtoon_df['Name']==selected_character)]['Sex'].values[0]
        character_keyword = webtoon_df[(webtoon_df['Title'] == selected_title) & (webtoon_df['Name']==selected_character)]['5_keywords'].values[0]

        col_1, col_2, col_3, col_4 = st.columns(4)
        with col_1:
            st.image(character_image, caption=f"{selected_title}의 {selected_character}")
        with col_2:
            st.markdown(f"- 이름   : {selected_character}")
            st.markdown(f"- 나이대 : {character_age}대")
            st.markdown(f"- 성별   : {character_sex}")
            st.markdown(f"- 키워드 : {character_keyword}")
    if st.session_state['user_1'][0]:
        with col_3:
            st.image(st.session_state['user_1'][0])
        with col_4:
            st.markdown(f"- 이름   : {st.session_state['user_1'][1]}")
            st.markdown(f"- 나이   : {st.session_state['user_1'][2]}")
            st.markdown(f"- 성별   : {st.session_state['user_1'][3]}")
            st.markdown(f"- 키워드 : {st.session_state['user_1'][4]}")

with st.container():
    _, _, col_middle, _, _ = st.columns(5)
    
    with col_middle:
        compare_button = st.button("비교시작", key='compare')
try:
    if compare_button:
        user_name = st.session_state['user_1'][1]
        user_age = st.session_state['user_1'][2]
        user_sex = st.session_state['user_1'][3]
        user_keyword = st.session_state['user_1'][4]

        # 이미지 변환
        user_image = Image.open(st.session_state['user_1'][0])
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
        st.image(concatenate_image, use_column_width='always', caption=f"\"{selected_character}와(과) {user_name} 비교 결과")

        radar_chart(round(age_score * 100, 2), round(keyword_score * 100, 2), round(image_score * 100, 2), sex_score * 100, round(total_score * 100, 2))
except:
    st.error("얼굴을 인식하지 못했습니다. 다른 사용자 사진을 업로드 해주세요.")