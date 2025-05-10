
#####################################---------------------------##############
###################👇below is with voice input and output
import streamlit as st
import google.generativeai as genai
import pyttsx3
import speech_recognition as sr
import os
# st.title("projrcts")
# Set your Gemini API key
# GEMINI_API_KEY = "AIzaSyBRTEq_pMAxzNvDK7tAz6m570J_vj43Gs4"
GEMINI_API_KEY = "AIzaSyBnmKPdqC4CeJ5U1lP_xfzWsjgWWzVjZ9E"

genai.configure(api_key=GEMINI_API_KEY)

# Set up pyttsx3 engine
engine = pyttsx3.init()

# Define plant-related keywords
PLANT_KEYWORDS = [
    "plant", "tree", "flower", "leaves", "stem", "roots", "photosynthesis", 
    "chlorophyll", "gardening", "agriculture", "horticulture", "plant disease", 
    "fungal infection", "bacterial infection", "viral disease", "pests", "plant care",
    "soil health", "watering", "fertilization", "plant nutrients", "plant growth"
]

def is_plant_related(prompt):
    """Check if the input text is related to plants and plant diseases."""
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in PLANT_KEYWORDS)

def get_plant_response(prompt):
    """Get response from Gemini API if the prompt is related to plants."""
    if is_plant_related(prompt):
        model = genai.GenerativeModel("gemini-1.5-pro")  # Using Gemini-Pro model
        response = model.generate_content(prompt)
        return response.text
    else:
        return "I only respond to plant-related queries. Please ask something about plants or plant diseases."

def generate_speech(text):
    """Convert the given text to speech and play it using pyttsx3."""
    engine.save_to_file(text, "response.mp3")
    engine.runAndWait()
    
    return "response.mp3"

# def recognize_speech():
#     """Capture speech input and convert it to text."""
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         st.info("Listening... Speak now!")
#         recognizer.adjust_for_ambient_noise(source)
#         try:
#             audio = recognizer.listen(source, timeout=10)
#             text = recognizer.recognize_google(audio)
#             return text
#         except sr.UnknownValueError:
#             return "Sorry, I couldn't understand your speech."
#         except sr.RequestError:
#             return "Could not request results. Check your internet connection."
def recognize_speech():
    """Capture speech input and convert it to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now!")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=10)
            text = recognizer.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            return "Listening timed out. You didn't speak in time."
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand your speech."
        except sr.RequestError:
            return "Could not request results. Check your internet connection."

# Streamlit UI
# st.set_page_config(page_title="Plant Care & Disease Chatbot", page_icon="🌱")

st.title("🌱 Plant Chatbot")
st.write("Ask me anything about plants, plant diseases, and plant care!")

# Initialize session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message(msg["role"]).markdown(f"**You:** {msg['text']}")
    else:
        st.chat_message(msg["role"]).markdown(f"**Bot:** {msg['text']}")

        # Add a button to generate speech for each bot response
        if st.button("🔊 Listen", key=msg["text"]):
            audio_file = generate_speech(msg["text"])
            st.audio(audio_file, format="audio/mp3")
            os.remove(audio_file)  # Clean up the temporary audio file

# Input field for user query with a microphone button
col1, col2 = st.columns([8, 1])

with col1:
    user_input = st.text_area("Enter your question:", st.session_state.get("voice_input", ""))

with col2:
    if st.button("🎤"):
        speech_text = recognize_speech()
        if "Sorry" not in speech_text:  # Only update if valid speech is recognized
            st.session_state["voice_input"] = speech_text
            st.rerun()

if st.button("Ask"):
    if user_input.strip():
        # Add user input to chat history
        st.session_state.messages.append({"role": "user", "text": user_input})
        
        # Get response from Gemini API
        response = get_plant_response(user_input)
        
        # Add bot response to chat history
        st.session_state.messages.append({"role": "bot", "text": response})
        ########################
        st.session_state["voice_input"] = ""  # Clear text area after asking 
        #############################
        # Re-render the chat history with the new messages
        st.rerun()
    else:
        st.warning("Please enter a question.")

# Footer
st.markdown("---")
st.markdown("🌿 *Powered by Gemini AI & Streamlit*")
