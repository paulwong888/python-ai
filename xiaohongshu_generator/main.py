from init_path import init
init()
import streamlit as st
from dotenv import load_dotenv
import os
from utils import generate_xiaohongshu
from commons.sidebar import display_sidebar, remove_button


st.title("爆款小紅書AI寫作助手 ✏️")
st.subheader("Present by Paul Wong")

openai_api_key = display_sidebar()
remove_button()
load_dotenv()

topic = st.text_input("主題", value="大模型")
submit = st.button("開始寫作")

if submit:
    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")
    if not topic:
        st.info("請輸入主題")
        st.stop()
    
    with st.spinner("AI正在努力创作中，請稍候..."):
        xiaohongshu_instance = generate_xiaohongshu(topic, openai_api_key)

    st.divider()

    left_column, right_column = st.columns(2)
    with left_column:
        titles = xiaohongshu_instance.titles
        for i in range(len(titles)):
            st.markdown(f"##### 小紅書標題{i+1}")
            st.write(titles[i])
    with right_column:
        st.markdown("##### 小紅書正文")
        st.write(xiaohongshu_instance.content)

