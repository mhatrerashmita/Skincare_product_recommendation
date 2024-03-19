# app.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import requests
from PIL import Image as PILImage
import streamlit as st

# Load dataset
@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('C:/Users/admin/PycharmProjects/indian_products/Skinpro - Skinpro.csv')
    return df

df = load_data()

# Concatenate 'Skin type' and 'Concern' into a single text column
df['text'] = df['Skin type'] + ' ' + df['Concern']

# Train the model
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])
text_clf.fit(df['text'], df['Product'])

# Function to predict all products related to user input
def predict_related_products(model, df, input_skin_type, input_concern):
    input_text = input_skin_type + ' ' + input_concern
    predicted_products = model.predict([input_text])

    matching_products = df[df['Product'].isin(predicted_products)]

    return matching_products

# Streamlit app
st.title('Skin Product Recommender')

# User input for skin type and concern
input_skin_type = st.text_input('Enter your skin type:')
input_concern = st.text_input('Enter your concern:')

if st.button('Predict'):
    related_products = predict_related_products(text_clf, df, input_skin_type, input_concern)

    if not related_products.empty:
        st.write("Images related to the given input:")
        for index, row in related_products.iterrows():
            response = requests.get(row['product_pic'], stream=True)
            if response.status_code == 200:
                img = PILImage.open(response.raw)
                img = img.resize((200, 200))
                st.image(img, caption=row['Product'], use_column_width=True)
            else:
                st.write(f"Failed to fetch image for product: {row['Product']}")
    else:
        st.write("No products found related to the given input.")
