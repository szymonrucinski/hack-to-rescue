import streamlit as st
from langchain.document_loaders import PyPDFLoader
from backend import get_problem_summary, get_text_summary, suggest
import logging
import json
import pandas as pd


# def add_bg_from_url():
#     page_bg_img = f"""
#     <style>
#     [data-testid="stAppViewContainer"] > .main {{
#     background-image: url("https://images.unsplash.com/photo-1501426026826-31c667bdf23d");
#     background-size: 180%;
#     background-position: top left;
#     background-repeat: no-repeat;
#     background-attachment: local;
#     }}

#     [data-testid="stSidebar"] > div:first-child {{
#     background-image: url("data:image/png;base64,{img}");
#     background-position: center;
#     background-repeat: no-repeat;
#     background-attachment: fixed;
#     }}

#     [data-testid="stHeader"] {{
#     background: rgba(0,0,0,0);
#     }}

#     [data-testid="stToolbar"] {{
#     right: 2rem;
#     }}
#     </style>
#     """


# st.markdown(page_bg_img, unsafe_allow_html=True)


# add_bg_from_url()

# create a title and sub-title
title_container = st.container()
col1, col2 = st.columns([50, 50])
# streamlit title with logo from url and text
with title_container:
    with col1:
        # st.title("UNDP solution finder")
        new_title = '<p style="font-family:Myriad Pro; color:Black; font-size:50px;text-align:center,font-weight:bold">SOLUTION GENERATOR</p>'
        st.markdown(new_title, unsafe_allow_html=True)
    with col2:
        st.image(
            "https://seeklogo.com/images/U/undp-logo-5682674D5C-seeklogo.com.png",
            width=300,
        )

# parse pdf file
uploaded_files = st.file_uploader("Choose a PDF file", accept_multiple_files=False)
if uploaded_files is not None:
    bytes_data = uploaded_files.read()
    # save bytes data to a file
    with open("file.pdf", "wb") as f:
        f.write(bytes_data)
    # st.write("filename:", uploaded_files.name)
    list_of_pages = []
    loader = PyPDFLoader("file.pdf")
    documents = loader.load()
    text_summary = get_text_summary(documents)
    st.text_area("Summary", text_summary, height=300)

    problems_summary = get_problem_summary(documents)
    problem_summary_json = json.loads(problems_summary)
    df = pd.DataFrame.from_dict(problem_summary_json, orient="index")
    st.table(df)

    st.write("Solution suggestions:")
    suggestion = suggest(problems_summary)
    for s in suggestion:
        df = pd.DataFrame.from_dict(s, orient="index")
        st.table(
            df,
        )
