import streamlit as st
from dotenv import load_dotenv
from utils import generate_script
import os

load_dotenv()

st.title("视频脚本生成器")
st.subheader("Present by Paul Wong")

with st.sidebar:
    openai_api_key = st.text_input("请输入OpenAi API密钥：", type="password")
    st.markdown("[获取OpenAi API密钥](https://platform.openai.com/account/api-key)")

subject = st.text_input("请输入视频的主题", value="sora模型")
video_length = st.number_input("请输入视频的大致时长(单位：分钟)", min_value=0.1, step=0.1, value=1.0)
creativity = st.slider(
    "请输入视频脚本的创造力(数字小说明更严重，数字大说明更多样)",
    min_value=0.1, max_value=1.0, value=0.2, step=0.1
)
submit = st.button("生成脚本")

if submit:
    if not openai_api_key:
        openai_api_key=os.getenv("OPENAI_API_KEY")
        # st.info("请输入OpenAi API密钥")
        # st.stop()
    if not subject:
        st.info("请输入视频的主题")
        st.stop()
    if not video_length >= 0.1:
        st.info("视频的长度需要大于或等于0.1")
        st.stop()

    with st.spinner(("AI正在思考中，请稍等...")):
        search_result, title, script = generate_script(subject, video_length, creativity, openai_api_key)
    st.success("视频脚本已生成！")
    st.subheader("标题： ")
    st.write(title)
    st.subheader("视频脚本： ")
    st.write(script)

    with st.expander("维基百科搜索结果 "):
        st.info(search_result)

    