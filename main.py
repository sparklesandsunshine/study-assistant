import base64
import re
import PyPDF2
import streamlit as st
from transformers import pipeline

# Streamlit
st.set_page_config(page_title= "Study Assistant", layout="wide")
st.title=("Study Assistant")
st.subheader("Your AI powered study companion")

 # Upload file
uploaded_file = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt"])

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")
    file_text = ""

    if uploaded_file.type == "application/pdf":
        uploaded_file.seek(0)
        b64 = base64.b64encode(uploaded_file.read()).decode('utf-8')
        st.markdown("### PDF Preview")
        st.markdown(f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="600px"></iframe>', unsafe_allow_html=True,)

        uploaded_file.seek(0)
        reader = PyPDF2.PdfReader(uploaded_file)
        for p in reader.pages:
            file_text += p.extract_text() or ""

    elif uploaded_file.type == "text/plain":
        file_text = uploaded_file.read().decode("utf-8")
        st.markdown("### Text Preview")
        st.text_area("Preview", file_text[:2000], height=300)
    
    file_text = re.sub(r"\n+", " ", file_text)
    file_text = re.sub(r"\s{2,}", " ", file_text).strip()

    if st.button("Summarize text"):
        with st.spinner("Summarizing..."):
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            chunk = file_text[:2000]
            result = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
            summary = result[0]["summary_text"]            
                
            st.subheader("Summary")
            st.write(summary)
            st.success("Summarization complete!")