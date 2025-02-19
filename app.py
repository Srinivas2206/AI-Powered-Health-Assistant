import streamlit as st
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.data.path.append("C:/Users/Srinivas M/AppData/Roaming/nltk_data")
# Download necessary NLTK data
#nltk.download('punkt')
nltk.download('stopwords')

# Load a pre-trained Hugging face model
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
    if "sneeze" in user_input or "sneezing" in user_input:
        return "Frequent sneezing may indicate allergies or a cold. Consult a doctor if symptoms persist."
    elif "symptom" in user_input:
        return "It seems like you're experiencing symtopms. please consult a doctor for accurate advice."
    elif "appointment" in user_input:
        return "Would you like me to schedule an appointment with a doctor?"
    elif "medication" in user_input:
        return "It's important to take your prescribed medications regularly. If you have concerns, conslut your doctor."
    else:
        context = """
        Common healthcare-related scenarios include symptoms of colds, flu, and allergies,
        along with medication guidance and appointment scheduling.
        """
        response = chatbot(question=user_input, context=context)
        return response['answer']
    

# Streamlit web app interface
def main():
    st.title("Healthcare Assistant Chatbot")
    user_input = st.text_input("How can I assist you today?","")
    if st.button("Submit"):
        if user_input:
            st.write("User: ", user_input)
            response = healthcare_chatbot(user_input)
            st.write("Healthcare Assistant: ", response)
        else:
            st.write("Please enter a query.")

if __name__ == "__main__":
    main()