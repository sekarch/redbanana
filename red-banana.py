import streamlit as st
import pandas as pd
import numpy as np

# Install necessary dependencies
st.subheader('Installing dependencies...')
st.code('!pip install scikit-learn pandas numpy')

# Import scikit-learn after installation
from sklearn.linear_model import Ridge

# Load your dataset
# Replace 'your_dataset.csv' with the actual file path or URL
data = pd.read_csv('Red_banana.csv')

# Assuming 'X' contains features and 'y' contains the target variable (Maturity Index)
X = data[['Fruit weight (g)', 'Fruit Length (cm)', 'Fruit Girth (cm)', 'Caliper (mm)']]
y = data['Maturity Index']

# Train Ridge Regression model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X, y)

# Background image URL
background_image_url = 'https://nrcb.icar.gov.in/album-diversity/Ruling%20Commercial%20Cultivars%20of%20India/Red%20Banana/slides/Red%20banana4.jpg'

# Streamlit app
st.markdown(
    f"""
    <style>
        body {{
            background-image: url("{background_image_url}");
            background-size: cover;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit components
st.title('Red Banana Maturity Predictor')

fruit_weight = st.number_input('Fruit Weight (g):')
fruit_length = st.number_input('Fruit Length (cm):')
fruit_girth = st.number_input('Fruit Girth (cm):')
caliper = st.number_input('Caliper (mm):')

if st.button('Predict'):
    # Make a prediction using the trained model
    input_data = np.array([[fruit_weight, fruit_length, fruit_girth, caliper]])
    maturity_index = ridge_model.predict(input_data)[0]

    # Display the result
    st.success(f'Maturity Index: {maturity_index}')
