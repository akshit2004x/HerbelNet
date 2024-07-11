import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np


def style_leaf_details():
    st.markdown(
        """
        <style>
        .leaf-details-container {
            padding: 10px;
            border-radius: 10px;
            background-color: rgb(14, 17, 23);
            margin-bottom: 10px;
            color: white;
            font-family: 'Be Vietnam Pro', sans-serif;
        }
        .leaf-category {
            color: rgb(46, 60, 86);
            text-align: left;
        }
        .leaf-content {
            color: rgb(218, 222, 229);
            text-align: right;
            margin-bottom: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

st.header('Leaf Classification Model')
model = load_model('Model\Final_model.keras')

medicinal_leaf_data = pd.read_csv('Data\MedicinalLeaf.csv')

leaf_categories = ['Aloevera', 'Amla', 'Amruthaballi', 'Arali', 'Astma_weed', 'Badipala', 'Balloon_Vine', 'Bamboo', 'Beans', 'Betel', 'Bhrami', 'Bringaraja', 'Caricature', 'Castor', 'Catharanthus', 'Chakte', 'Chilly', 'Citron lime (herelikai)', 'Coffee', 'Common rue(naagdalli)', 'Coriender', 'Curry', 'Doddpathre', 'Drumstick', 'Ekka', 'Eucalyptus', 'Ganigale', 'Ganike', 'Gasagase', 'Ginger', 'Globe Amarnath', 'Guava', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jackfruit', 'Jasmine', 'Kambajala', 'Kasambruga', 'Kohlrabi', 'Lantana', 'Lemon', 'Lemongrass', 'Malabar_Nut', 'Malabar_Spinach', 'Mango', 'Marigold', 'Mint', 'Neem', 'Nelavembu', 'Nerale', 'Nooni', 'Onion', 'Padri', 'Palak(Spinach)', 'Papaya', 'Parijatha', 'Pea', 'Pepper', 'Pomoegranate', 'Pumpkin', 'Raddish', 'Rose', 'Sampige', 'Sapota', 'Seethaashoka', 'Seethapala', 'Spinach1', 'Tamarind', 'Taro', 'Tecoma', 'Thumbe', 'Tomato', 'Tulsi', 'Turmeric', 'ashoka', 'camphor', 'kamakasturi', 'kepala']

img_height = 224
img_width = 224
image = st.file_uploader('Upload Leaf Image', type=['jpg', 'png'])

if image is not None:
    image_load = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_arr = np.expand_dims(img_arr, axis=0)

    predict = model.predict(img_arr)

    predicted_class_index = np.argmax(predict)
    predicted_leaf_category = leaf_categories[predicted_class_index]


    leaf_details = medicinal_leaf_data[medicinal_leaf_data['LeafName'] == predicted_leaf_category]

    st.image(image_load, width=200)
    st.markdown('<div class="leaf-details-container">', unsafe_allow_html=True)
    st.markdown('<div class="leaf-category">Predicted Leaf Category: ' + predicted_leaf_category + '</div>', unsafe_allow_html=True)

    if not leaf_details.empty:
        style_leaf_details()  # Apply styling
        st.markdown('<div class="leaf-table">', unsafe_allow_html=True)
        for index, row in leaf_details.iterrows():
            for column, value in row.items():
                st.markdown(f"<div><span class='leaf-category'>{column}:</span> <span class='leaf-content'>{value}</span></div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="leaf-content">No details found for the predicted leaf category.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
