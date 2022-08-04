import streamlit as st
import pandas as pd
import numpy as np
import cv2
import PIL
import plotly.express as px
from PIL import Image

from image_modules import image_similarity, image_visualization, convert_image_to_anime
from text_modules import text_similarity

@st.cache(allow_output_mutation=True)
def load_model_keyword():
    model_keyword = text_similarity.Text()
    return model_keyword

@st.cache(allow_output_mutation=True)
def load_model_rs50():
    model_rs50 = image_similarity.Res50()
    return model_rs50

@st.cache(allow_output_mutation=True)
def load_model_animeGan():
    model_animeGan = convert_image_to_anime.AnimeGAN()
    return model_animeGan

@st.cache(allow_output_mutation=True)
def load_model_orb():
    model_orb = image_visualization.ORB()
    return model_orb

@st.cache(allow_output_mutation=True)
def load_webtoon_df():
    webtoon_df = pd.read_excel('./text_data/Webtoon_keywords_test.xlsx')
    return webtoon_df

def radar_chart(age_score, keyword_score, image_score, rank=0, famous=0):
    df = pd.DataFrame(dict(
    r=[age_score,
       keyword_score,
       image_score,
       rank,
       famous],
    theta=['나이','키워드 유사도','이미지 유사도',
           '랭킹', '유명도']))
    fig = px.line_polar(df, r='r', theta='theta', line_close=True, range_r =(0, 100))
    st.write(fig)

if 'session' not in st.session_state:
    st.session_state['session'] = 0
    
if 'character_list' not in st.session_state:
    st.session_state['character_list'] = []

if 'recommend_n' not in st.session_state:
    st.session_state['recommend_n'] = 0

if 'upload_image' not in st.session_state:
    st.session_state['upload_image'] = 0

if 'anime_image' not in st.session_state:
    st.session_state['anime_image'] = 0

model_keyword = load_model_keyword()
model_rs50 = load_model_rs50()
model_animeGan = load_model_animeGan()
model_orb = load_model_orb()

webtoon_df = load_webtoon_df()

st.title("나와 어울리는 웹툰 등장인물은??")

with st.form('form_1', clear_on_submit=False):
    upload_image_bytes = st.file_uploader("1. 사진을 업로드하세요.", type='jpg', accept_multiple_files=False)
    upload_image_name = st.text_input("2. 사진 이름을 입력하세요.", placeholder="여기에 입력하세요.")
    upload_image_age = st.number_input("3. 나이를 입력하세요.", min_value=0, max_value=120, value=25)
    upload_image_sex = st.radio("4. 성별을 고르세요.", ('남', '여'), horizontal=True)
    upload_image_keyword = st.text_input("5. 키워드를 입력하세요.", placeholder="구분자는 ', '입니다.")
    character_n = st.slider("6. 확인할 등장인물의 수를 설정하세요!!", 0, len(webtoon_df), 0)
    submitted_1 = st.form_submit_button('완료')

try:
    with st.container():
        if submitted_1:
            upload_image = Image.open(upload_image_bytes)
            st.session_state['upload_image'] = upload_image
            anime_user_image = model_animeGan.run(upload_image.convert("RGB"))
            st.session_state['anime_image'] = anime_user_image
            st.session_state['recommend_n'] = character_n

            character_list = []

            filtered_webtoon_df = webtoon_df[webtoon_df["Sex"]==upload_image_sex].reset_index(drop=True)

            recommend_bar = st.progress(0)
            for idx in range(len(filtered_webtoon_df)):
                recommend_bar.progress((idx + 1) / len(filtered_webtoon_df))

                character_dict = dict()

                webtoon_title = filtered_webtoon_df['Title'][idx]
                character_name = filtered_webtoon_df['Name'][idx]
                character_age = filtered_webtoon_df['Age'][idx]
                character_sex = filtered_webtoon_df['Sex'][idx]
                character_keyword = filtered_webtoon_df['5_keywords'][idx]

                try:
                    character_image = Image.open(f"./image_data/Webtoon/{webtoon_title}_{character_name}.jpg")
                except:
                    continue

                if upload_image_age > character_age:
                    age_score = character_age / upload_image_age
                else:
                    age_score = upload_image_age / character_age
                keyword_score = model_keyword.get_keyword_similarity(upload_image_keyword, character_keyword)
                image_score = model_rs50.run(anime_user_image, character_image)
                total_score = (keyword_score * 0.2) + (image_score * 0.7) + (age_score * 0.1)

                character_dict["Title"] = webtoon_title
                character_dict["Name"] = character_name
                character_dict["Age"] = character_age
                character_dict["Sex"] = character_sex
                character_dict["Keyword"] = character_keyword
                character_dict["Age_score"] = age_score
                character_dict["Keyword_score"] = keyword_score
                character_dict["Image_score"] = image_score
                character_dict["Total_score"] = total_score

                character_list.append(character_dict)

            character_list.sort(key=lambda x : x['Total_score'], reverse=True)
            st.session_state['character_list'] = character_list
except:
    st.error("업로드하신 사진의 얼굴이 인식되지 않습니다. 다른 사진을 사용해주세요.")

if st.session_state['character_list']:
    with st.form('form_2', clear_on_submit=True):
            form_2_col_1, form_2_col_2 = st.columns(2)
            with form_2_col_1:
                submitted_2 = st.form_submit_button('이전')
            with form_2_col_2:
                submitted_3 = st.form_submit_button('다음')
            if submitted_2 and int(st.session_state['session']) > 0:
                st.session_state['session'] -= 1
            elif submitted_3 and (int(st.session_state['session']) < st.session_state['recommend_n'] - 1):
                st.session_state['session'] += 1
                
    character_list = st.session_state["character_list"]
    webtoon_title = character_list[st.session_state["session"]]["Title"]
    character_name = character_list[st.session_state["session"]]["Name"]
    character_age = character_list[st.session_state["session"]]["Age"]
    character_sex = character_list[st.session_state["session"]]["Sex"]
    character_keyword = character_list[st.session_state["session"]]["Keyword"]
    age_score = character_list[st.session_state["session"]]["Age_score"]
    keyword_score = character_list[st.session_state["session"]]["Keyword_score"]
    image_score = character_list[st.session_state["session"]]["Image_score"]
    total_score = character_list[st.session_state["session"]]["Total_score"]
    
    cv2_upload_image = cv2.cvtColor(np.array(st.session_state['upload_image']), cv2.COLOR_RGB2BGR)
    cv2_upload_image = cv2.resize(cv2_upload_image, (256, 256))
    cv2_upload_image = cv2.cvtColor(cv2_upload_image, cv2.COLOR_RGB2BGR)
    character_image = cv2.imread(f"./image_data/Webtoon/{webtoon_title}_{character_name}.jpg", flags=cv2.IMREAD_COLOR)
    anime_user_image = model_rs50.convert_type(st.session_state['anime_image'], TYPE=1, GRAY=False)
    orb_knn_image = model_orb.run(anime_user_image, character_image, show=False)
    
    concatenate_image = np.concatenate([cv2_upload_image, orb_knn_image], axis=1)
    st.image(concatenate_image, use_column_width='always', caption=f"\"{webtoon_title}\"의 {character_name}와(과) {upload_image_name} 비교 결과")
    
    with st.container():
        col_1, col_2 = st.columns(2)
        
        with col_1:
            st.image(cv2.cvtColor(character_image, cv2.COLOR_BGR2RGB), caption=f"{webtoon_title}의 {character_name}", use_column_width='always')
        with col_2:
            rank = int(st.session_state["session"]) + 1
            st.markdown(f"### < {rank}위 등장인물 정보 >")
            st.markdown(f"- 웹툰   : {webtoon_title}")
            st.markdown(f"- 이름   : {character_name}")
            st.markdown(f"- 나이   : {character_age}대")
            st.markdown(f"- 성별   : {character_sex}")
            st.markdown(f"- 키워드 : {character_keyword}")
        
        radar_chart(round(age_score * 100, 2), round(keyword_score * 100, 2), round(image_score * 100, 2))