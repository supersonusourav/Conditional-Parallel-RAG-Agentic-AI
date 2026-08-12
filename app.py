import streamlit as st
from graph import app
from state import State

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="College Assistant AI - Agentic RAG",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for Olive Green Theme
st.markdown("""
<style>
    /* Main Background & Base Text */
    .stApp {
        background-color: #556B2F; /* Olive Green */
        color: #FFFFFF;
    }
    
    /* Header Styling */
    .main-title {
        color: #F0E68C; /* Khaki / Light Gold Accent */
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #E6E6FA; /* Light Lavender Accent */
        font-size: 1.1rem;
        margin-bottom: 25px;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #2b331f;
        color: #d1d8c5;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label {
        color: #E2E8F0 !important;
    }

    /* Chat Messages Container */
    .stChatMessage[data-testid="stChatMessage"] {
        background-color: #3B4A23; /* Darker Olive for Messages */
        border-radius: 10px;
        color: #FFFFFF;
        margin-bottom: 10px;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #4A5D2C; /* Slightly Lighter Olive Alternate */
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #6B8E23; /* Olive Drab Button */
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #8FBC8F; /* Dark Sea Green Hover */
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🎓 Student Profile")
    st.markdown("---")
    
    programme = st.selectbox(
        "Select Your Programme:",
        ["BCA", "BBA", "B.Com (H)"],
        index=0
    )
    
    st.markdown("---")
    st.info(f"**Current Status:** Session active for **{programme}**")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<h1 class="main-title">College Assistant AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Ask anything about your courses, exams, fees, or general campus information.</p>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": f"Hello! I am your College Assistant. How can I help you with your **{programme}** queries today?"}
    ]

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------------------------------------------------------
# User Input & Execution Pipeline
# -----------------------------------------------------------------------------
if prompt := st.chat_input("Type your question here..."):
    # Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display Assistant Response with Spinner
    with st.chat_message("assistant"):
        with st.spinner("Processing request..."):
            try:
                initial_state: State = {
                    "programme": programme,
                    "messages": [("human", prompt)],
                    "categories": [],
                    "retrieved_contexts": {}
                }

                # Invoke LangGraph Application
                result = app.invoke(initial_state)
                response_text = result["messages"][-1].content

                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

            except Exception as e:
                error_msg = f"An error occurred while generating a response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})