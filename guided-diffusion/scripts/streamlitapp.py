import streamlit as st
import pandas as pd
import numpy as np
from classifier_sample import main

st.title('Uber pickups in NYC')

if st.button('Generated Images'):
    main(batch_size=4, num_samples=5, timestep_respacing=250, )