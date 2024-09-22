from init_path import init
init()
import streamlit as st
from dotenv import load_dotenv
from utils import generate_script
from commons.sidebar import display_sidebar, remove_button
import os

openai_api_key = display_sidebar()
remove_button()
load_dotenv()

st.title("🎬 視頻腳本生成器")
st.subheader("Present by Paul Wong")


subject = st.text_input("💡 請輸入視頻的主題", value="sora模型")
video_length = st.number_input("⏱️ 請輸入視頻的大致時長(組織：分鐘)", min_value=0.1, step=0.1, value=1.0)
creativity = st.slider(
    "✨ 請輸入視頻腳本的創造力(數位小說明更嚴重，數位大說明更多樣)",
    min_value=0.1, max_value=1.0, value=0.2, step=0.1
)
submit = st.button("生成腳本")

if submit:
    if not openai_api_key:
        openai_api_key=os.getenv("OPENAI_API_KEY")
        # st.info("请输入OpenAi API密钥")
        # st.stop()
    if not subject:
        st.info("請輸入視頻的主題")
        st.stop()
    if not video_length >= 0.1:
        st.info("視頻的長度需要大於或等於0.1")
        st.stop()

    with st.spinner(("AI正在思考中，請稍等...")):
        search_result, title, script = generate_script(subject, video_length, creativity, openai_api_key)
    st.success("視頻腳本已生成！")
    st.subheader("🔥 标题： ")
    st.write(title)
    st.subheader("📝 視頻腳本： ")
    st.write(script)

    with st.expander("維基百科搜索結果 👀"):
        st.info(search_result)

    