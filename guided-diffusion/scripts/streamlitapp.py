import streamlit as st
import pandas as pd
import numpy as np
from classifier_sample import main
# st.set_option('browser.serverAddress', 'localhost')
st.title('Diffusion Model - Generates Chest Xray!')



if st.button('Generate 3 Images'):

    col1, col2, col3 = st.columns(3)
    cols_list = [col1, col2, col3]
    path = main()
    array = np.load(path)
    labels = array["arr_1"]
    label_dict = {0:"No Pleural Effusion", 1: "Pleural Effusion"}
    for idx, img in enumerate(array["arr_0"]):
        with cols_list[idx]:
            st.header(label_dict[labels[idx]])
            st.image(img)

st.subheader("OR")
number = st.text_input('Generate infinite images')
# st.write('The current number is ')

if number:
    if st.button(f'Generate {number} Images'):
        path = main(int(number))
        array = np.load(path)
        labels = array["arr_1"]
        label_dict = {0:"No Pleural Effusion", 1: "Pleural Effusion"}
        for idx, img in enumerate(array["arr_0"]):
            st.image(img, caption=label_dict[labels[idx]])