import streamlit as st 
import pandas as pd

json_data = pd.read_json('datasets/imagenet_class_index.json')

mini_json = {
    "17": ["n01580077", "jay"]
}

st.title('Welcome to my Machine learning')
st.write('Time to build a machine learning project')

tab1, tab2 = st.tabs(['Zero', 'One']) 

with tab1:
    st.json(mini_json)


with tab2:
    username = st.session_state.name
    age = st.slider("What's your age", min_value=10, max_value=100)
    text = st.text_area('What is all this about?')
    
    st.write(text)
    st.text_input('Enter your name', key='name')
    st.write('Username:  ' + username)
    st.write(f'Age: {age}')