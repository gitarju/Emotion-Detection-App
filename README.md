# Facial Emotion Detection  

Hey there! Welcome to my deep learning mini-project. 

This project is a complete end-to-end AI pipeline that detects human faces in an image, figures out what emotion they're expressing, and then actually responds to you with a contextual sentence based on how you're feeling. It's built as a **Streamlit Web Application** so you can easily deploy it and interact with it in your browser!

## What does it do?
1. **Face Detection**: Uses OpenCV's Haar Cascades to find faces in an uploaded image.
2. **Emotion Classification**: Uses a custom Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify the emotion into one of 7 categories (Happy, Sad, Angry, Fear, Surprise, Disgust, or Neutral).
3. **NLP Responses**: Uses NLTK to spit out a fun, context-aware response based on the detected emotion.

## The Dataset
To train the brain behind this project, I used the **FER-2013 (Facial Expression Recognition 2013)** dataset. It consists of thousands of 48x48 pixel grayscale images of faces, all centered and standardized.

You can grab the dataset yourself from Kaggle right here:  
👉 [FER-2013 Dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

## How to run locally
I designed this to be super easy to run. No messy terminal commands or complex deployment setups.

1. Clone this repository.
2. Install the requirements: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run app.py`
4. Open the link it gives you in your browser, upload an image, and watch the AI do its thing!

## How to Deploy to Streamlit Community Cloud
1. Push this entire repository to your GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New app"**, select your repository, set the main file path to `app.py`, and click **Deploy**!

## Tech Stack
* **Python** (The glue holding it all together)
* **Streamlit** (For the beautiful and easy-to-use web interface)
* **TensorFlow / Keras** (Deep learning model architecture and training)
* **OpenCV** (Computer Vision & Face cropping)
* **NLTK** (Natural Language Processing for the bot's replies)

Feel free to fork this, train the model for more epochs, or swap out the NLP responses with your own funny quotes. Enjoy! ✌️
