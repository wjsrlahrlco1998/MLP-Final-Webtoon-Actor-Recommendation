import streamlit as st
import pandas as pd

def add_feedback_session():
    st.session_state['Feedback_id'] += 1

st.sidebar.markdown("## 😍User Feedback😍")
st.sidebar.markdown("""사용자 여러분의 피드백을 반영하여 더 개선되고 다양한 기능으로 찾아올 수 있도록 노력하겠습니다.

**사용자 여러분의 다양한 의견을 들려주세요!!**""")
    
with st.form("feedback_form", clear_on_submit=True):
    st.markdown("### 당신의 의견을 들려주세요!!")
    user_evaluate = st.slider("1. 해당 서비스에 대한 별점을 매겨주세요.", min_value=0.0, max_value=5.0, value=2.5, step=0.5)
    user_feedback_text = st.text_area("2. 자유로운 의견을 들려주세요!!", max_chars=256, value="", placeholder="자유로운 의견 부탁드립니다!")
    submmit = st.form_submit_button('의견제출', on_click=add_feedback_session)

    if submmit and user_evaluate:
        st.success("성공적으로 제출되었습니다!")
        df = pd.read_csv("./text_data/User_Feedback.csv", encoding='utf-8-sig')
        df = df.append({"id":st.session_state["Feedback_id"], "stars" : user_evaluate, "opinion" : user_feedback_text}, ignore_index = True)
        df.to_csv("./text_data/User_Feedback.csv", encoding='utf-8-sig', index=False)
        submmit=False