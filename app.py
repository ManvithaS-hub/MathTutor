# app.py
import streamlit as st
from main import math_precheck_llm_agent  # adjust this import

st.set_page_config(page_title="Math Tutor", layout="wide")
st.title("Your Math Guru")

query = st.text_input("Enter your math question:")

if query:
    with st.spinner("Thinking..."):
        response = math_precheck_llm_agent(query)

    st.markdown("Question")
    st.write(query)

    st.markdown("Answer")
    st.write(response)

