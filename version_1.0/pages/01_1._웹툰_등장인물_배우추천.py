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

# Define radar chart function
def radar_chart(age_score, keyword_score, image_score, harmony, sex_score=100):
    df = pd.DataFrame(dict(
    r=[age_score,
       keyword_score,
       image_score,
       harmony,
       sex_score],
    theta=['나이','키워드 유사도','이미지 유사도',
           '조화', '성별']))
    fig = px.line_polar(df, r='r', theta='theta', line_close=True, range_r =(0, 100))
    st.write(fig)

# Handling session state
def init_session():
    st.session_state['session'] = 0
    
# Load AI Model from session
model_keyword = st.session_state['model_keyword']
model_AE = st.session_state['model_AE']
model_animeGan = st.session_state['model_animeGan']
model_orb = st.session_state['model_orb']

# Load Data from session
actor_df = st.session_state['actor_df']
webtoon_df = st.session_state['webtoon_df']
keyword_score_df = st.session_state['keyword_score_df']
webtoon_titles = list(set(webtoon_df['Title'].unique().tolist()))
webtoon_titles.sort()

st.markdown("## 🕵️‍♂️웹툰 등장인물과 어울리는 배우 찾기")
st.sidebar.markdown("# 🛠기능설명")
st.sidebar.markdown("**웹툰 등장인물 추천 기능**은 데이터베이스에 등록된 **70개의 웹툰**과 **1746개의 배우 데이터**를 이용하여 해당 웹툰의 등장인물에 가장 어울리는 배우를 이미지, 키워드, 나이, 조화, 성별의 지표를 토대로 이를 종합한 점수로 추천해주는 기능입니다.")
st.sidebar.markdown("# 📋사용설명")
st.sidebar.markdown("""1. 원하는 웹툰 제목을 선택하세요.
2. 원하는 등장인물을 선택하세요.
3. 추천받을 배우의 나이 범위를 지정해주세요.
4. 추천받을 배우의 수를 지정해주세요.
5. '1'~'5'번의 과정을 모두 마쳤으면 **'배우추천'** 버튼을 클릭해주세요.
6. '이전'과 '다음'버튼으로 추천된 배우를 확인할 수 있습니다.""")


# Page
with st.form('form_1', clear_on_submit=False):
    selected_title = st.selectbox("1. 원하는 웹툰 제목을 검색하세요!!", webtoon_titles, key="sbox1", disabled=False)
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
    if submitted_3:
        st.success(f"추천받을 배우의 나이가 {min_age}세에서 {max_age}세로 설정되었습니다.")

with st.form('form_4', clear_on_submit=False):
    filtered_actor_df = actor_df[(actor_df['Sex'] == character_sex) & (actor_df['Age'] >= min_age) & (actor_df['Age'] <= max_age)]
    filtered_actor_df = filtered_actor_df.reset_index(drop=True)
    actor_n = st.number_input(f"4. 추천받을 배우의 수를 설정하세요!! (MAX : {len(filtered_actor_df)})", min_value = 0, max_value = len(filtered_actor_df), value = 10)
    submitted_4 = st.form_submit_button('완료')
    if submitted_4:
        st.success(f"추천받을 배우의 수가 {actor_n}명으로 설정되었습니다.")

with st.container():
    _, _, middle, _, _ = st.columns(5)

    with middle:
        recommend_button = st.button("배우 추천", key='recommend', on_click=init_session)

if recommend_button:
    st.session_state['character_image_1'] = character_image
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
        image_score = (model_AE.run(actor_image, st.session_state['character_image_1']) - 0.7) / 0.3
        if image_score < 0:
            image_score = 0

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
                
    orb_control = st.radio("매칭 지점확인", ["On", 'Off'], horizontal=True, index = 1)

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
    img = st.session_state['character_image_1']
    cv2_character_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    if orb_control == "On":
        orb_knn_image = model_orb.run(cv2_actor_image, cv2_character_image)
    elif orb_control == "Off":
        orb_knn_image = model_orb.run(cv2_actor_image, cv2_character_image, ratio=0.0)

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

        radar_chart(round(age_score * 100, 2), round(keyword_score * 100, 2), round(image_score * 100, 2), round(total_score * 100, 2))

else:
    st.write("추천된 배우가 없습니다.")
    
# except:
#    st.error("해당 등장인물은 사진이 존재하지 않습니다.")