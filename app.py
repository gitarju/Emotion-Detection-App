import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import random
import nltk
from nltk.tokenize import word_tokenize

# --- NLP Responses Setup ---
# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

EMOTION_RESPONSES = {
    'angry': ["Take a deep breath. Count to 10.", "It looks like you're frustrated."],
    'disgust': ["Yuck! Something doesn't look right."],
    'fear': ["It's okay to feel scared. You are safe here."],
    'happy': ["You look highly positive today! Keep it up!", "Your smile brightens the room!"],
    'neutral': ["You're completely calm and composed.", "A perfectly balanced emotional state."],
    'sad': ["I'm sorry you are feeling down.", "Don't be sad! Everything will be okay."],
    'surprise': ["Whoa! What just happened? You look shocked!"]
}

def generate_nlp_response(emotion_label, confidence):
    if confidence < 50.0:
        return "I'm not quite sure how you feel right now. My confidence is low.", []
    responses = EMOTION_RESPONSES.get(emotion_label, ["I see a face..."])
    selected_sentence = random.choice(responses)
    try:
        tokens = word_tokenize(selected_sentence)
    except:
        tokens = selected_sentence.split()
    return selected_sentence, tokens

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
COLOR_MAP = {
    'happy': (0, 255, 0), 'sad': (255, 0, 0), 'angry': (0, 0, 255), 
    'surprise': (0, 255, 255), 'fear': (255, 0, 255), 
    'disgust': (0, 100, 0), 'neutral': (255, 255, 255)
}

# --- Streamlit UI ---
st.set_page_config(page_title="Emotion Detection AI", page_icon="🤖")

st.title("Facial Emotion Detection & NLP AI 🚀")
st.write("Upload a picture of a face, and the AI will predict the emotion and respond to you!")

# --- Load Models (Cached for performance) ---
@st.cache_resource
def load_models():
    model_path = 'models/emotion_model.h5'
    cascade_path = os.path.join('haarcascade', 'haarcascade_frontalface_default.xml')
    
    if not os.path.exists(model_path):
        st.error(f"Cannot find model at {model_path}.")
        return None, None
    if not os.path.exists(cascade_path):
        st.error(f"Cannot find Haar Cascade at {cascade_path}.")
        return None, None
        
    model = tf.keras.models.load_model(model_path)
    face_classifier = cv2.CascadeClassifier(cascade_path)
    return model, face_classifier

model, face_classifier = load_models()

if model and face_classifier:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        image = Image.open(uploaded_file).convert('RGB')
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) == 0:
            st.warning("No faces detected in the image.")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        else:
            # We process faces
            results = []
            for (x, y, w, h) in faces:
                roi_gray = gray_frame[y:y+h, x:x+w]
                roi_resized = cv2.resize(roi_gray, (48, 48))
                roi_norm = roi_resized.astype('float') / 255.0
                roi_tensor = np.expand_dims(np.expand_dims(roi_norm, axis=-1), axis=0)
                
                preds = model.predict(roi_tensor, verbose=0)[0]
                label_index = preds.argmax()
                confidence = preds[label_index] * 100
                label_text = EMOTION_LABELS[label_index]
                box_color = COLOR_MAP.get(label_text, (255, 255, 255))
                
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
                cv2.putText(frame, f"{label_text.capitalize()} ({confidence:.1f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
                
                nlp_response, _ = generate_nlp_response(label_text, confidence)
                results.append({
                    "emotion": label_text.capitalize(),
                    "confidence": confidence,
                    "response": nlp_response
                })
            
            # Convert back to RGB for displaying in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="Processed Image", use_column_width=True)
            
            st.subheader("AI Analysis")
            for i, res in enumerate(results):
                st.write(f"**Face {i+1}**: {res['emotion']} ({res['confidence']:.1f}%)")
                st.info(f"🤖 AI Assistant says: {res['response']}")
