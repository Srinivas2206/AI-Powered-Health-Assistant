import streamlit as st
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data (better portability)
nltk.download('punkt')
nltk.download('stopwords')

# Load a pre-trained Hugging Face model
chatbot = pipeline("question-answering", model="deepset/bert-base-cased-squad2")

# Preprocess user input
def preprocess_input(user_input):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(user_input)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Define healthcare-specific response logic
def healthcare_chatbot(user_input):
    user_input = preprocess_input(user_input).lower()
    
    # Rule-based responses
    if "sneeze" in user_input or "sneezing" in user_input:
        return "Frequent sneezing may indicate allergies or a cold. Consult a doctor if symptoms persist."
    elif "symptom" in user_input:
        return "It seems like you're experiencing symptoms. Please consult a doctor for accurate advice."
    elif "appointment" in user_input:
        return "Would you like me to schedule an appointment with a doctor?"
    elif "medication" in user_input:
        return "It's important to take your prescribed medications regularly. If you have concerns, consult your doctor."
    
    # Context-based AI response
    context = """
    Common healthcare-related scenarios include symptoms of colds, flu, and allergies,
    along with medication guidance and appointment scheduling.
    """
    try:
        response = chatbot(question=user_input, context=context)
        return response['answer']
    except Exception as e:
        return "I'm sorry, I couldn't process your request. Please try again."

# Streamlit web app interface
def main():
    st.title("🩺 Healthcare Assistant Chatbot")
    
    user_input = st.text_area("How can I assist you today?", "")
    
    if st.button("Submit"):
        if user_input.strip():
            response = healthcare_chatbot(user_input)
            st.write("**Healthcare Assistant:**", response)
        else:
            st.warning("⚠️ Please enter a query.")

if __name__ == "__main__":
    main()
