import streamlit as st

st.set_page_config(page_title= "Study Assistant", layout="wide")
st.title=("Study Assistant")
st.subheader("Your AI powered study companion")
st.write("""
Welcome to your personal Study Assistant!      
         
**What it can do:
- Provide a summary of your notes
- Extract key ideas and topics
- Generate flashcards 
- Help you study smarter not harder

- Lets begin with uploading your file            
""")

uploaded_file = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt"])

if uploaded_file is not None:
    st.success(f"You uploaded: {uploaded_file.name}")
    st.write("File processing will go here...")



