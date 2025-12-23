import streamlit as st
import cv2
import numpy as np

from mediapipe_utils import extract_landmarks
from model import model, SIGNS
from labels import SIGN_LABELS

st.set_page_config(page_title="Sign Language Translator")
st.title("🤟 Sign Language Recognition")
st.subheader("Output in Hindi & Manipuri")

camera_image = st.camera_input("Capture hand gesture")

if camera_image is not None:
    bytes_data = camera_image.getvalue()
    np_img = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    landmarks = extract_landmarks(frame)

    input_data = np.expand_dims(np.expand_dims(landmarks, axis=0), axis=0)
    probs = model.predict(input_data)

    sign_key = SIGNS[np.argmax(probs)]

    hindi = SIGN_LABELS[sign_key]["hindi"]
    manipuri = SIGN_LABELS[sign_key]["manipuri"]

    cv2.putText(frame, f"Sign: {sign_key}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    st.success(f"🟠 Hindi: {hindi}")
    st.success(f"🔵 Manipuri: {manipuri}")



     

      



