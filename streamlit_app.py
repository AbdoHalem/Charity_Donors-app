import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App')
st.info('This is app predicts the annual salary of a person whether is is more than 50K dollars or less !')

with st.expander("Data"):
    st.write('**Raw Data**')
    data = pd.read_csv("cesus.csv")
    data
