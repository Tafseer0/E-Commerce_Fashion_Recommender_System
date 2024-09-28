import streamlit as st
import os
from PIL import Image
import pickle
import numpy as np
from numpy.linalg import norm
import tensorflow
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50 ,preprocess_input
from sklearn.neighbors import NearestNeighbors

st.title('E-Commerce Fashion Recommender System')

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
#print(feature_list.shape)
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


def feature_extraction(image_path,model):
    img = image.load_img(image_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / np.linalg.norm(result)
    return normalized_result

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def recommendation(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors= 6, algorithm='brute',metric='euclidean').fit(feature_list)
    neighbors.fit(feature_list)

    distances,indices=neighbors.kneighbors([features])
    return indices

#step 1: upload image ->save
#load image ->feature extraction ->recommendation
#recommendation 
#display

uploaded_file = st.file_uploader("Choose an image", type="jpg")
if uploaded_file is not None:
   if save_uploaded_file(uploaded_file):
       #display the image
       display_image = Image.open(uploaded_file)
       st.image(display_image, caption="Uploaded Image", width=350)
       #feature extraction
       features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
       #    st.text(features)
       #reccomendation
       indices = recommendation(features,feature_list)
       #show the recommendation
       col1, col2, col3, col4, col5 = st.columns(5)

       with col1:
           st.image(filenames[indices[0][1]], caption="Recommend 1",use_column_width=True)

       with col2:
           st.image(filenames[indices[0][2]], caption="Recommend 2",use_column_width=True)
       
       with col3:
           st.image(filenames[indices[0][3]], caption="Recommend 3",use_column_width=True)

       with col4:
           st.image(filenames[indices[0][4]], caption="Recommend 4",use_column_width=True)

       with col5:
           st.image(filenames[indices[0][5]], caption="Recommend 5",use_column_width=True)

   else:
        st.header("Some error occurred in file upload")

