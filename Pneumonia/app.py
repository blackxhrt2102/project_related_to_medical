import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
import keras 
from streamlit_lottie import st_lottie
from keras.preprocessing.image import load_img,img_to_array 
import numpy as np
import tensorflow as tf
import streamlit as st
import time
from PIL import Image,ImageOps
import json

def load_lottiefile(filepath: str):
  with open(filepath, "r") as f:
    return json.load(f)

def processing(image,model):
  image=np.asarray(image)
  image_dims=np.expand_dims(image,axis=0)
  res=model.predict(image_dims)
  result=np.argmax(res)
  return result
lottie_coding = load_lottiefile("girl-cycling-in-autumn.json")
st.title('Agri-EI 🍀')
st.markdown('**Disease Prediction for plants:-**')
st.text('Made by Rahul Jha')

st.text('         ')
st.text('         ')

selected=option_menu(
            menu_title=None,  # required
            options=['About the problem 🖥️','Practical👨‍💻','Hire me👨‍🎓'],  # required
            icons=["home", "book", "envelope"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "orange"},
            },
        )

st.text('   ')
st.text('   ')
st.text('   ')
st.text('   ')
if(selected=='About the problem 🖥️'):
  st.header('About')
  st.image('images.jpg')
  st.markdown(""" **Pneumonia is an infection that inflames the air sacs in one or both lungs. It kills more children younger than 5 years old each year than any other infectious disease, such as HIV infection, malaria, or tuberculosis.
   Diagnosis is often based on symptoms and physical examination. Chest X-rays may help confirm the diagnosis.**""")

  st.markdown('**-----------------------------------------------------------------------------------------------**')
  st.header('Dataset')
  st.image('cnn1.png')
  st.markdown("""**This dataset contains 5,856 validated Chest X-Ray images. The images are split into a training set and a testing set of independent patients. Images are labeled as (disease:NORMAL/BACTERIA/VIRUS)-(randomized patient ID)-(image number of a patient). For details of the data collection and description, see the referenced paper below.

According to the paper, the images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou.

A previous version (v2) of this dataset is available here: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia. 
Note that the files names are irregular in v2, but they are fixed in the new version (v3).**""")


  st.markdown('**#------------------------------------------------------------------------------------------------**')  


#-----------------------------------------------------------------------------------------------------



elif(selected=='Practical👨‍💻'):
  st.header('Model Prediction')
  st.subheader('Please upload the image of xray of lungs')
  st.text('    ')
  st.text('    ')
  st.text('    ')
  classes=['Pneumonia','Healthy']
  with st.spinner('Uploading the model for prediction'):
    model=tf.keras.models.load_model('Potato.h5')
    if model is not None:
      time.sleep(5)
      st.subheader('Please upload the image of xray of lungs:-')
      image_path=st.file_uploader('Upload the image here',type=['jpeg','png','jpg'])
    if image_path is not None:
      img=Image.open(image_path)
      img=ImageOps.fit(img,(150,150),Image.ANTIALIAS)
      result=processing(img,model)
      result=classes[result]
      with st.spinner('Analysing the picture'):
        time.sleep(5)
        st.image(image_path,caption=result)
        st.markdown('**The name of the type is:-**')
        st.balloons()
        st.subheader(result)



    




#---------------------------------------------------------------------

elif(selected=='Hire me👨‍🎓'):
  st.header('About me')
  lottie_coding2=load_lottiefile("80680-online-study.json")
  st_lottie(
    lottie_coding2,
    speed=1,
    reverse=False,
    loop=True,
    quality="low", 
    height=700,
    width=600,
    key=None,
)
  st.markdown("""**I have recently graduate in 2021 and was not able to get a job in campus placement.So for last 3 months I have being reworking 
  for my passion in data science.I am currently doing internship upder Almabetter.**""")
  st.text(' ')
  st.text('  ')
  st.markdown('**A chance  would be appreciated 🙏**')
  st.text(' ')
  st.subheader('Contact:-')
  st.markdown('**Phone ☎️ :- 6202239544**')
  st.markdown('**Email 📧 :- rahuljha0610@gmail.com**')
  st.markdown('**Linkedin 🚦 :- [link](https://www.linkedin.com/in/rahul-jha-600047164/)**')
  
