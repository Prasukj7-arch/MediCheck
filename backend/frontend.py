import streamlit as st
import requests

# Center the title and description using columns
st.markdown("<h1 style='text-align: center;'>Agentic AI Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask questions about Agentic AI based on the V-Soft Consulting blog.</p>", unsafe_allow_html=True)

# Initialize session state for query history
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# FastAPI endpoint
API_URL = "http://localhost:8000"

# Sidebar for query history
with st.sidebar:
    st.subheader("Query History")
    if st.session_state.query_history:
        for item in st.session_state.query_history:
            st.markdown(f"**{item['type']} Query:** {item['query']}")
            st.markdown(f"**Response:** {item['response']}")
            st.markdown("---")
    else:
        st.write("No queries yet.")

# Main content centered using columns
col1, col2, col3 = st.columns([1, 2, 1])  # Middle column wider for centering
with col2:
    st.subheader("Ask a Question")
    question = st.text_input("Enter your question:", key="qna_input")
    if st.button("Submit Question"):
        if question.strip():
            try:
                response = requests.post(f"{API_URL}/qna", json={"text": question})
                response.raise_for_status()
                result = response.json()
                st.markdown("**Question:**")
                st.write(result["question"])
                st.markdown("**Answer:**")
                st.write(result["answer"])
                st.session_state.query_history.append({"type": "Q&A", "query": result["question"], "response": result["answer"]})
            except requests.RequestException as e:
                st.error(f"Error: {e}")
        else:
            st.error("Please enter a valid question.")