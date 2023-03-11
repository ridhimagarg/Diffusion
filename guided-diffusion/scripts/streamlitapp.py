import streamlit as st
import pandas as pd
import numpy as np
from classifier_sample import main
# st.set_option('browser.serverAddress', 'localhost')
st.title('Diffusion Model - Generates Chest Xray!')


st.header("Synthetic Images")
col1, col2, col3 = st.columns(3)
cols_list = [col1, col2, col3]
array = np.load("/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertResults/classifiersample/openai-2022-12-05-17-46-00-565034_5000samples3attempt10classifierscalelatermodels85k/samples_5000x256x256x3.npz")

images = [array["arr_0"][9], array["arr_0"][11], array["arr_0"][16]]
# labels = ["No Pleural Effusion", "No Pleural Effusion", ]

for idx, img in enumerate(images):
    with cols_list[idx]:
        st.image(img)


st.header("Real Images")
col1, col2, col3 = st.columns(3)
cols_list = [col1, col2, col3]
array = np.load("/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertResults/evaluations_base/5000_256_256_3_reference_batch.npz")

images = [array["arr_0"][9], array["arr_0"][11], array["arr_0"][16]]
for idx, img in enumerate(images):
    with cols_list[idx]:
        st.image(img)



if st.button('Generate 3 Images'):

    col1, col2, col3 = st.columns(3)
    cols_list = [col1, col2, col3]
    path = main()
    array = np.load(path)
    labels = array["arr_1"]
    label_dict = {0:"Pleural Effusion", 1: "No Pleural Effusion"}
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
        label_dict = {0:"Pleural Effusion", 1: "No Pleural Effusion"}
        for idx, img in enumerate(array["arr_0"]):
            st.image(img, caption=label_dict[labels[idx]])