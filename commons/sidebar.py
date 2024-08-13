import streamlit as st

def display_sidebar():
    with st.sidebar:
        openai_api_key = st.text_input("请输入OpenAI API密钥： ", type="password")
        st.markdown("[获取OpenAi API密钥](https://platform.openai.com/account/api-key)")
    return openai_api_key

def remove_button():
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