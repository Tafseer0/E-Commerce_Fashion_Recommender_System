# E-commerce-Fashion-Recommender-System
This project is a fashion recommendation system built using Convolutional Neural Networks (CNN) and TensorFlow. It helps users find fashion items similar to those they upload, enhancing the shopping experience by providing personalized and relevant product suggestions.

The e-commerce fashion recommender system analyzes an image of a product uploaded by the user and provides recommendations for similar items from the product catalog. This system is designed to be efficient, accurate, and scalable for real-time use on e-commerce platforms.

project Working: https://drive.google.com/file/d/1Z-TUyGDvUiNFvlirEvzg2pUuVuJhNcT7/view?usp=drive_link

Features
Image-based Recommendations: Upload a fashion product image and receive visually similar recommendations.
Deep Learning Model: Utilizes CNN for image feature extraction and TensorFlow for model training and inference.
Large Dataset: Trained on a dataset of 25 GB of fashion images to ensure high accuracy.
Scalable for E-commerce: The system is optimized for deployment on fashion e-commerce websites.

Dataset
Size: 25 GB of fashion images.
Preprocessing: Images were resized and normalized for input into the CNN model. Augmentation techniques such as flipping, rotation, and scaling were used to improve model generalization.
Source: The dataset consists of labeled images from various fashion categories.
Link for Dataset = https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

Problem Faced during the project
1. Tensorflow version used <2.11 with numpy 1.26.4 i wested so much time to fix this version error between tensorflow and python.
2. Tensorflow_gpu setup
3. Streamlit port setup when i try to run streamlit faced port permission error so i change default port to 5958.
   this can also work if anyone dont want to change the default port then use this "streamlit run app_name.py --server.port 5998"
4.  
