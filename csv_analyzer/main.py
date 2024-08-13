import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from utils import dataframe_agent
from streamlit.runtime.uploaded_file_manager import UploadedFile

load_dotenv()

def create_char(response_dict):
    chart_type = response_dict["chart_type"]
    input_data = response_dict["datas"]
    df_data = pd.DataFrame(input_data["data"], columns=input_data["columns"])
    df_data.set_index(input_data["columns"][0], inplace=True)
    if chart_type == "bar":
        st.bar_chart(df_data)
    elif chart_type == "line":
        st.line_chart(df_data)
    if chart_type == "scatter":
        st.scatter_chart(df_data)

st.title("CSV数据分析智能工具")
st.subheader("Present by Paul Wong")

with st.sidebar:
    openai_api_key = st.text_input("请输入OpenAI API密钥： ", type="password")
    st.markdown("[获取OpenAi API密钥](https://platform.openai.com/account/api-key)")

data = st.file_uploader("上传你的数据文件 (CSV格式)", type="csv")
sample_data_button = st.button("使用样例数据文件")

if data:
    print(type(data))
    st.session_state["df"] = pd.read_csv(data)
elif sample_data_button:
    sample_data = "csv_analyzer/personal_data.csv"
    st.session_state["df"] = pd.read_csv(sample_data)

if "df" in st.session_state:
    with st.expander("原始数据"):
        st.dataframe(st.session_state["df"])

query = st.text_area(
    "请输入你关于以上表格的问题，或数据提取请求，或可视化要求(支持散点图，折线图，条形图)， 如/绘制出职业的条形图/绘制出客户年收入和年龄之间的散点图",
    value="请提取年龄大于30的数据"
)
button = st.button("生成回答")

if button:
    if not openai_api_key:
        openai_api_key=os.getenv("OPENAI_API_KEY")
    if "df" not in st.session_state:
        st.info("请先上传数据文件")
        st.stop()
    if not query:
        st.info("请输入要统计的问题")
        st.stop()
    if "df" in st.session_state:
        with st.spinner("AI正在思考中，请稍等..."):
            response_dict = dataframe_agent(openai_api_key, st.session_state["df"], query)
            if "answer" in response_dict:
                st.write(response_dict["answer"])
            elif "table" in response_dict:
                st.table(
                    pd.DataFrame(
                        response_dict["table"]["data"],
                        columns=response_dict["table"]["columns"]
                    )
                )
            else:
                create_char(response_dict)
        