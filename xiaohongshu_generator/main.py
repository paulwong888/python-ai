import streamlit as st
from dotenv import load_dotenv
import os
from utils import generate_xiaohongshu

from utils import generate_xiaohongshu

load_dotenv()
# Add custom CSS to hide the GitHub icon
st.markdown(
    r"""
    <style>
    #MainMenu {visibility: hidden;}
    [data-testid="stActionButton"] {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, 
    unsafe_allow_html=True
)

st.title("爆款小红书AI写作助手 ✏️")
st.subheader("Present by Paul Wong")

with st.sidebar:
    openai_api_key = st.text_input("请输入OpenAI API密钥：", type="password")
    st.markdown("[获取OpenAI API密钥](https://platform.openai.com/account/api-key)")

topic = st.text_input("主题", value="大模型")
submit = st.button("开始写作")

if submit:
    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")
    if not topic:
        st.info("请输入主题")
        st.stop()
    
    with st.spinner("AI正在努力创作中，请稍候..."):
        xiaohongshu_instance = generate_xiaohongshu(topic, openai_api_key)

    st.divider()

    left_column, right_column = st.columns(2)
    with left_column:
        titles = xiaohongshu_instance.titles
        for i in range(len(titles)):
            st.markdown(f"##### 小红书标题{i+1}")
            st.write(titles[i])
    with right_column:
        st.markdown("##### 小红书正文")
        st.write(xiaohongshu_instance.content)

