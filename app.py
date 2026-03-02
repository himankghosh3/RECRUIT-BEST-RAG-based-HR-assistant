import streamlit as st
import ollama
from data import load_dataset, get_top_k_chunks

df = load_dataset()


st.title("RECRUIT-best")
st.header("Make the best choice regarding your recruitment drive")

with st.expander('Preview dataset'):
    st.dataframe(df)


req = st.text_input("Enter your requirements:")

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


