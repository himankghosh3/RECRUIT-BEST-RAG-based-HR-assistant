import streamlit as st
import ollama
from data import load_dataset, get_top_k_chunks

st.set_page_config(page_title="RECRUIT-best", page_icon=":briefcase:", layout="wide")

df = load_dataset()


st.title("Recruit-Best: AI Hiring Assistant | Powered by RAG ")
st.caption("Made by Himank Ghosh")
st.markdown("Welcome to Recruit-Best, your AI-powered hiring assistant! This tool uses **Retrieval-Augemented-Generation (RAG)** , Allowing you to make informed decisions during your recruitment drive by analyzing candidate data and providing insights based on your requirements.")
st.markdown("---")

st.subheader("Candidate Dataset")
with st.expander("Click to view candidate dataset"):
    st.dataframe(df, use_container_width=True)

st.markdown("---")

st.subheader("Ask Recruit-Best")

col1, col2 = st.columns([2,1])

with col1:
    req = st.text_input("Enter your requirements:")

with col2:
    st.info("Example: 'Find candidates with experience in Python and data analysis.'")

if req:
    top_chunks = get_top_k_chunks(req, k=3)
    context = "\n".join(top_chunks)
    
    response = ollama.chat(
        model='phi',
        messages=[
            {'role': 'system', 'content': f"Use the following candidate data to help answer the user's query:\n\n{context}"},
            {'role': 'user', 'content': req}
        ]
    )

     
    answer = response['message']['content'] 
    st.write(answer) 

st.markdown("---")
st.markdown("Made by Himank Ghosh")


