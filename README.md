# Multiclass-Fish-Image-Classification
🐟 Fish Classification App

A Streamlit-based web app that classifies fish images into multiple categories using a trained TensorFlow/Keras deep learning model. The app provides visual confidence scores for each class using interactive Plotly charts.

🔹 Features

Classifies fish images into 11 categories, including both freshwater and seafood fish.

Displays prediction confidence for all classes with a horizontal bar chart.

Handles multi-class and binary outputs robustly.

Suppresses TensorFlow, Python, and Streamlit warnings for a clean user experience.

Fully interactive and user-friendly with Streamlit.

🐠 Fish Categories
Category	Description
animal_fish	General fish category
animal_fish_bass	Bass fish
fish_sea_food_black_sea_sprat	Black Sea Sprat
fish_sea_food_gilt_head_bream	Gilt-head bream
fish_sea_food_hourse_mackerel	Horse mackerel
fish_sea_food_red_mullet	Red mullet
fish_sea_food_red_sea_bream	Red sea bream
fish_sea_food_sea_bass	Sea bass
fish_sea_food_shrimp	Shrimp
fish_sea_food_striped_red_mullet	Striped red mullet
fish_sea_food_trout	Trout

🛠️ Technology Stack

Python 3.10+

TensorFlow / Keras – for model loading and predictions

Streamlit – for the web interface

Plotly – interactive visualization of prediction confidence

Pillow (PIL) – image processing

NumPy & Pandas – data manipulation

🧠 Model

Pre-trained Keras model (.keras) stored in the models folder.

Input images are resized to 224x224 pixels and normalized before prediction.

Supports both multi-class softmax and single-output sigmoid models.

📂 Repository Structure
fish-classifier/
│
├─ app.py                  # Main Streamlit app
├─ models/
│   └─ best_fish_model.keras   # Trained Keras model
├─ requirements.txt        # Python dependencies
└─ README.md               # Project overview

⚡ Future Improvements

Add real-time webcam classification.

Integrate more fish species for a larger dataset.

Add history/logging of uploaded images and predictions.

Deploy on Streamlit Cloud or Heroku for online access.
