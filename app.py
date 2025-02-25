import streamlit as st
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import datetime
from gtts import gTTS
import os
import random

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load a pre-trained question-answering model
chatbot = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Preprocess user input
def preprocess_input(user_input):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(user_input)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Text-to-speech function using gTTS
def speak_text(text):
    tts = gTTS(text=text, lang="en")
    tts.save("response.mp3")
    os.system("start response.mp3")  

# Doctor appointment scheduling
def schedule_appointment():
    now = datetime.datetime.now()
    appointment_date = now + datetime.timedelta(days=1)
    return f"Your appointment is scheduled for {appointment_date.strftime('%Y-%m-%d at %I:%M %p')}."

# Daily health tip
def health_tips():
    tips = [
        "Drink at least 8 glasses of water daily.",
        "Get 7-9 hours of sleep every night.",
        "Exercise for at least 30 minutes per day.",
        "Eat a balanced diet with fruits and vegetables.",
        "Manage stress through meditation or deep breathing.",
        "Limit processed foods and opt for whole foods instead.",
        "Maintain good posture to prevent back and neck pain.",
        "Avoid excessive screen time and take breaks.",
    ]
    return random.choice(tips)

# Medication reminder
def set_medication_reminder():
    return "Reminder set! Please take your medication on time."

# Emergency Contact Suggestion
def emergency_contact():
    return "In case of emergency, call 112 or visit the nearest hospital immediately."

# BMI Calculator
def calculate_bmi(weight, height):
    bmi = weight / (height ** 2)
    if bmi < 18.5:
        return f"Your BMI is {bmi:.2f}. You are underweight. Consider consulting a nutritionist."
    elif 18.5 <= bmi < 24.9:
        return f"Your BMI is {bmi:.2f}. You have a healthy weight. Keep maintaining your lifestyle!"
    elif 25 <= bmi < 29.9:
        return f"Your BMI is {bmi:.2f}. You are overweight. Consider regular exercise and a balanced diet."
    else:
        return f"Your BMI is {bmi:.2f}. You are in the obese category. Consult a doctor for guidance."

# Healthcare chatbot response
def healthcare_chatbot(user_input):
    processed_input = preprocess_input(user_input).lower()

    keyword_responses = {
        "sneeze": "Frequent sneezing may indicate allergies or a cold. Consult a doctor if symptoms persist.",
        "symptom": "It seems like you're experiencing symptoms. Please consult a doctor for accurate advice.",
        "appointment": schedule_appointment(),
        "medication": set_medication_reminder(),
        "tip": health_tips(),
        "emergency": emergency_contact(),
        "bmi": "To calculate BMI, please enter your weight (kg) and height (m)."
    }

    for key, response in keyword_responses.items():
        if key in processed_input:
            return response

    context = """
    Common healthcare-related scenarios include symptoms of colds, flu, and allergies,
    along with medication guidance and appointment scheduling.
    """
    try:
        response = chatbot(question=user_input, context=context)
        return response.get("answer", "I'm sorry, I couldn't process your request. Please try again.")
    except Exception:
        return "I'm sorry, I couldn't process your request. Please try again."

# Streamlit web app interface
def main():
    st.title("🩺 AI-Powered Healthcare Assistant")
    st.write("**Daily Health Tip:**", health_tips())
    
    user_input = st.text_area("How can I assist you today?", "")
    
    if st.button("Submit"):
        if user_input.strip():
            response = healthcare_chatbot(user_input)
            st.write("**Healthcare Assistant:**", response)
            speak_text(response)
        else:
            st.warning("⚠️ Please enter a query.")
    
    st.header("Calculate Your BMI")
    weight = st.number_input("Enter your weight (kg)", min_value=1.0, format="%.2f")
    height = st.number_input("Enter your height (m)", min_value=0.5, format="%.2f")
    
    if st.button("Calculate BMI") and weight > 0 and height > 0:
        st.write(calculate_bmi(weight, height))

if __name__ == "__main__":
    main()
