# Package Load
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import PIL
import plotly.express as px
import io
from PIL import Image
from keras.preprocessing import image as keras_image

# AI Model Module Load
from image_modules import AutoEncoder_2D, image_visualization, convert_image_to_anime
from text_modules import text_similarity

def image_to_byte_array(img):
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='jpeg')
    img_byte = img_byte_arr.getvalue()
    return img_byte

# Load AI Model from session
model_animeGan = st.session_state['model_animeGan']

st.markdown("## 👼내가 웹툰 캐릭터가 된다면??")
st.sidebar.markdown("# 🛠기능설명")
st.sidebar.markdown("**웹툰 캐릭터 변환 기능**은 내가 올린 사진을 웹툰의 그림체로 바꾸어주는 기능입니다.")
st.sidebar.markdown("# 📋사용설명")
st.sidebar.markdown("""1. 나의 사진을 업로드하세요.
2. '변환'버튼을 누르세요.
3. 변환된 사진은 '다운로드'버튼을 통해서 다운로드할 수 있습니다. """)

with st.form('form_upload', clear_on_submit=False):
    user_image_bytes = st.file_uploader("1. 나의 사진을 업로드하세요.", type=['jpg', 'png'], accept_multiple_files=False)
    convert_type = st.radio("2. 변환모드를 선택하세요.", ('Basic', 'Webtoon(Beta)'), horizontal=True)
    submitted = st.form_submit_button("변환하기")
try:
    if submitted and (convert_type=='Basic'):
            user_image = Image.open(user_image_bytes)
            user_anime_image = model_animeGan.run(user_image.convert("RGB"))

            st.session_state['5_user_image'][0] = user_image
            st.session_state['5_user_image'][1] = user_anime_image
    elif submitted and (convert_type=='Webtoon(Beta)'):
        st.error("해당 기능은 준비중 입니다.")
except:
    st.error("얼굴을 인식하지 못했습니다. 다른 사진을 사용해주세요.")

col_1, col_2 = st.columns(2)

if st.session_state['5_user_image'][0]:
    with col_1:
        st.markdown("### <변환전>")
        st.image(st.session_state['5_user_image'][0].convert('RGB'))
    with col_2:
        st.markdown("### <변환후>")
        st.image(st.session_state['5_user_image'][1])
    
    _, _, middle, _, _ = st.columns(5)
    
    result = image_to_byte_array(st.session_state['5_user_image'][1])
    with middle:
        st.download_button("다운로드", data=result, file_name="My_Character.jpg", mime="image/jpg")