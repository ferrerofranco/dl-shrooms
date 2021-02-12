from fastai.vision.all import *
import streamlit as st
from PIL import Image
import numpy as np

learn = load_learner(Path('export.pkl'))

st.write("""
         # Poisonous/Edible Mushroom Prediction
         # **DO NOT** USE IN REAL LIFE SCENARIO
         """
         )
st.write("This is an PoC on transforming models into web apps. The accuracy of the model is not the main focus,**DO NOT** use it as a real life mushroom classifier")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
        
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    image = PILImage.create(np.asarray(image))
    tfms = Normalize.from_stats(*imagenet_stats)
    image = tfms(image)
    st.image(image, use_column_width=True)
    pred_class,_,_ = learn.predict(image)
    
    st.write('''It is {}!  
    **PREDICTION NOT ACCURATE PLEASE DO *NOT* CONSUME THE MUSHROOM**'''.format(pred_class))