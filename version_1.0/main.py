import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import time
from keras.preprocessing import image as keras_image
import plotly.express as px

from image_modules import image_similarity, image_visualization

@st.cache(allow_output_mutation=True)
def load_actor_df():
    actor_df = pd.read_excel('./text_data/Actor_keywords_test.xlsx')
    return actor_df

@st.cache(allow_output_mutation=True)
def load_webtoon_df():
    webtoon_df = pd.read_excel('./text_data/Webtoon_keywords_test.xlsx')
    return webtoon_df

@st.cache(allow_output_mutation=True)
def load_keyword_score_df():
    keyword_score_df = pd.read_excel('./text_data/Keyword_score_table_test.xlsx')
    return keyword_score_df

@st.cache(allow_output_mutation=True)
def load_model_rs50():
    model_rs50 = image_similarity.Res50()
    return model_rs50

@st.cache(allow_output_mutation=True)
def load_model_orb():
    model_orb = image_visualization.ORB()
    return model_orb

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

if 'actor_list' not in st.session_state:
    st.session_state['actor_list'] = []

if 'recommend_n' not in st.session_state:
    st.session_state['recommend_n'] = 0
    
actor_df = load_actor_df()
webtoon_df = load_webtoon_df()
keyword_score_df = load_keyword_score_df()
webtoon_titles = webtoon_df['Title'].unique().tolist()

model_rs50 = load_model_rs50()
model_orb = load_model_orb()

# streamlit Title
st.title("Test")

with st.form('form_1', clear_on_submit=False):
    selected_title = st.selectbox("1. 원하는 웹툰 제목을 검색하세요!!", set(webtoon_titles), key="sbox1", disabled=False)
    submitted_1 = st.form_submit_button('완료')

with st.form('form_2', clear_on_submit=False):
    character_list = webtoon_df[webtoon_df['Title']==selected_title]['Name'].to_list()
    selected_character = st.radio("2. 원하는 등장인물을 선택하세요!!", set(character_list))
    submitted_2 = st.form_submit_button('완료')

with st.container():        
    if submitted_2 or selected_character:
        character_image = Image.open(f"./image_data/Webtoon/{selected_title}_{selected_character}.jpg")
        character_age = webtoon_df[(webtoon_df['Title'] == selected_title) & (webtoon_df['Name']==selected_character)]['Age'].values[0]
        character_sex = webtoon_df[(webtoon_df['Title'] == selected_title) & (webtoon_df['Name']==selected_character)]['Sex'].values[0]
        character_keyword = webtoon_df[(webtoon_df['Title'] == selected_title) & (webtoon_df['Name']==selected_character)]['5_keywords'].values[0]

        col_1, col_2 = st.columns(2)
        with col_1:
            st.image(character_image, caption=f"{selected_title}의 {selected_character}", width = 256)
        with col_2:
            st.markdown("### < 등장인물 정보 >")
            st.markdown(f"- 이름   : {selected_character}")
            st.markdown(f"- 나이대 : {character_age}대")
            st.markdown(f"- 성별   : {character_sex}")
            st.markdown(f"- 키워드 : {character_keyword}")

with st.form('form_3', clear_on_submit=False):
    min_age, max_age = st.slider("3. 추천받을 배우의 나이 범위를 지정하세요!!", 0, 100, (10, 20))
    submitted_3 = st.form_submit_button('완료')

with st.form('form_4', clear_on_submit=False):
    filtered_actor_df = actor_df[(actor_df['Sex'] == character_sex) & (actor_df['Age'] >= min_age) & (actor_df['Age'] <= max_age)]
    filtered_actor_df = filtered_actor_df.reset_index(drop=True)
    actor_n = st.slider("4. 추천받을 배우의 수를 설정하세요!!", 0, len(filtered_actor_df), 0)
    submitted_4 = st.form_submit_button('완료')

with st.container():
    _, _, middle, _, _ = st.columns(5)
    
    with middle:
        recommend_button = st.button("배우 추천", key='recommend')
        
if recommend_button:
    recommend_bar = st.progress(0)
    actor_list = []
    
    for idx in range(len(filtered_actor_df)):
        recommend_bar.progress((idx + 1) / len(filtered_actor_df))
        
        actor_dict = dict()
        actor_name = filtered_actor_df['Name'][idx]
        actor_age = filtered_actor_df['Age'][idx]
        actor_sex = filtered_actor_df['Sex'][idx]
        actor_keyword = filtered_actor_df['5_keywords'][idx]
        
        try:
            actor_image = Image.open(f"./image_data/Actor2webtoon/webtoon_{actor_name}_{actor_age}_{actor_sex}.jpg")
        except:
            actor_image = Image.open(f"./image_data/Actor/{actor_name}_{actor_age}_{actor_sex}.jpg")
        
        keyword_score = keyword_score_df[(keyword_score_df["Title"]==selected_title) &
                                        (keyword_score_df["Name"]==selected_character)][f"{actor_name}_{actor_age}_{actor_sex}"].values[0]
        image_score = model_rs50.run(actor_image, character_image)
        if actor_age > character_age:
            age_score = character_age / actor_age
        else:
            age_score = actor_age / character_age
        total_score = (keyword_score * 0.3) + (image_score * 0.7)
        
        actor_dict["Name"] = actor_name
        actor_dict["Age"] = actor_age
        actor_dict["Sex"] = actor_sex
        actor_dict["Keyword"] = actor_keyword
        actor_dict["Age_score"] = age_score
        actor_dict["Keyword_score"] = keyword_score
        actor_dict["Image_score"] = image_score
        actor_dict["Total_score"] = total_score
        
        actor_list.append(actor_dict)

    actor_list.sort(key=lambda x : x['Total_score'], reverse=True)
    st.session_state['recommend_n'] = actor_n
    st.session_state["actor_list"] = actor_list
    
if st.session_state["actor_list"]:
    with st.form('form_5', clear_on_submit=True):
            form_5_col_1, form_5_col_2 = st.columns(2)
            with form_5_col_1:
                submitted_5 = st.form_submit_button('이전')
            with form_5_col_2:
                submitted_6 = st.form_submit_button('다음')
            if submitted_5 and int(st.session_state['session']) > 0:
                st.session_state['session'] -= 1
            elif submitted_6 and (int(st.session_state['session']) < st.session_state['recommend_n'] - 1):
                st.session_state['session'] += 1
            
    actor_list = st.session_state["actor_list"]
    actor_name = actor_list[st.session_state["session"]]["Name"]
    actor_age = actor_list[st.session_state["session"]]["Age"]
    actor_sex = actor_list[st.session_state["session"]]["Sex"]
    actor_keyword = actor_list[st.session_state["session"]]["Keyword"]
    age_score = actor_list[st.session_state["session"]]["Age_score"]
    keyword_score = actor_list[st.session_state["session"]]["Keyword_score"]
    image_score = actor_list[st.session_state["session"]]["Image_score"]
    total_score = actor_list[st.session_state["session"]]["Total_score"]
    
    r_actor_image = cv2.imread(f"./image_data/Actor/{actor_name}_{actor_age}_{actor_sex}.jpg", flags=cv2.IMREAD_COLOR)
    r_actor_image = model_orb.img_resize(r_actor_image)
    r_actor_image = model_orb.img_to_rgb(r_actor_image)
    
    try:
        _ = Image.open(f"./image_data/Actor2Webtoon/webtoon_{actor_name}_{actor_age}_{actor_sex}.jpg")
        cv2_actor_image = cv2.imread(f"./image_data/Actor2Webtoon/webtoon_{actor_name}_{actor_age}_{actor_sex}.jpg", flags=cv2.IMREAD_COLOR)
    except:
        cv2_actor_image = cv2.imread(f"./image_data/Actor/{actor_name}_{actor_age}_{actor_sex}.jpg", flags=cv2.IMREAD_COLOR)
    cv2_character_image = cv2.cvtColor(np.array(character_image), cv2.COLOR_RGB2BGR)
    
    orb_knn_image = model_orb.run(cv2_actor_image, cv2_character_image)
    
    concatenate_image = np.concatenate([r_actor_image, orb_knn_image], axis=1)
    st.image(concatenate_image, use_column_width='always', caption=f"\"{selected_title}\"의 {selected_character}와(과) {actor_name} 비교 결과")
    
    with st.container():
        r_col_1, r_col_2 = st.columns(2)
        
        with r_col_1:
            st.image(r_actor_image, caption=f"배우 {actor_name}", use_column_width='always')
            
        with r_col_2:
            rank = int(st.session_state["session"]) + 1
            st.markdown(f"### < {rank}위 배우 정보 >")
            st.markdown(f"- 이름   : {actor_name}")
            st.markdown(f"- 나이   : {actor_age}")
            st.markdown(f"- 성별   : {actor_sex}")
            st.markdown(f"- 키워드 : {actor_keyword}")
        
        radar_chart(round(age_score * 100, 2), round(keyword_score * 100, 2), round(image_score * 100, 2))
        
else:
    st.write("추천된 배우가 없습니다.")