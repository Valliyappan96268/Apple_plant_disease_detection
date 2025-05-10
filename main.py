import streamlit as st
import streamlit as st
from PIL import Image
st.set_page_config(
    page_title="Multipage app",
    page_icon="👻",
)
# st.title("Main Page")

st.sidebar.success("Select a page above. ")

# ✅ Initialize session state safely
# if "start" not in st.session_state:
#     st.session_state.start = False
# Set page configuration

# Load plant image
plant_image = Image.open("wallpaper.jpg")  # Replace with your image path

# Title Section
st.title("🍎 Apple Plant Disease Analyzer")
st.subheader("Early Detection, Better Protection")
st.markdown("""
Welcome to the **Plant Disease Analyzer** — your smart assistant for detecting plant diseases through leaf images.
This AI-powered tool helps farmers and gardeners keep plants healthy and productive.
""")

# Display plant image and features side by side
col1, col2 = st.columns([1, 2])

with col1:
    st.image(plant_image, caption="Healthy Plant", use_container_width=True)

with col2:
    st.markdown("### 🔍 Key Features")
    st.markdown("""
    - 📷 Upload plant leaf images for instant disease detection  
    - 🧠 Powered by Deep Learning (R-CNN)  
    - 💬 Smart treatment recommendations using AI  
    - 📊 Easy-to-read analysis  
    - 🗣️ Use AI chatbot for clarify the questions  
    """)

    st.markdown("### 🚀 Ready to protect your plants?")
    # st.button("Start Detection", use_container_width=True)
    # if st.button("Start Detection", use_container_width=True):
    #     st.session_state.start = True
    with st.expander("ℹ️Analysis your Plant"):
        st.markdown("""
        
        1. click the Analyser in side bar.  
        2. Upload a clear image of a plant leaf.  
        3. You receive instant diagnosis and suggested remedies.  
        """)

# Footer or expandable info section
with st.expander("ℹ️ Chat bot"):
    st.markdown("""
    1. Click the Chatbot in side bar.  
    2.Ask your's question by text or voice.  
    3. Get the responce like text and audio.  
    """)

# Footer note
st.markdown("---")
st.markdown("📌 Developed with ❤️ for precision agriculture.")

# if st.session_state.start:
#     st.switch_page("pages/app.py")
