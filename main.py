import base64  # Encode binary to base64 for PDF preview iframe
import csv  # CSV utilities (reserved for future export/use)
import io  # In-memory byte/stream utilities (reserved for future use)
import nltk  # Natural Language Toolkit for tokenization
import yake  # Keyword extraction library
import re  # Regular expressions for text cleaning
import PyPDF2  # PDF reader to extract text from PDFs
import streamlit as st  # Streamlit web UI framework
from transformers import pipeline  # Hugging Face pipeline for summarization
from nltk.tokenize import sent_tokenize  # Sentence tokenizer
import random  # Random utilities (e.g., shuffling)
random.seed(42)  # Make behavior deterministic where randomness is used

# Cache heavy resources (model) so they load once per session
@st.cache_resource
def get_summarizer():  # Returns a cached summarization pipeline
    return pipeline("summarization", model="facebook/bart-large-cnn")

for resource in ["punkt", "punkt_tab"]:  # Ensure required NLTK tokenizers are available
    try:
        nltk.data.find(f"tokenizers/{resource}")  # Check if resource is already installed
    except LookupError:
        nltk.download(resource)  # Download missing resource at runtime

# --- Helpers ---------------------------------------------------------------
def extract_keywords(text, max_k=10):  # Extract up to max_k keywords using YAKE
    kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=max_k)  # Configure YAKE for unigram keywords
    kws = [k for k, score in kw_extractor.extract_keywords(text)]  # Run extractor and keep only keyword strings

    seen, clean = set(), []  # Track seen keywords and the final cleaned list
    for k in kws:  # Iterate over extracted keywords
        k2 = k.strip().strip(".,:;!?").lower()  # Normalize: trim punctuation and lowercase
        if k2 and k2 not in seen and len(k2) > 2:  # Filter out empties, duplicates, and very short tokens
            seen.add(k2)  # Mark keyword as seen
            clean.append(k2)  # Add to results
    return clean  # Return cleaned keywords

def make_cloze_cards(text, keywords, max_cards=8):  # Build cloze (fill-in-the-blank) flashcards
    sents = sent_tokenize(text)  # Split text into sentences
    cards = []  # Store resulting (cloze_sentence, answer) tuples
    for s in sents:  # Consider each sentence
        hit = next((k for k in keywords if k in s.lower()), None)  # Find first keyword present in sentence
        if not hit:  # If no keyword in sentence
            continue  # Skip sentence
        pattern = hit  # The substring to mask in the sentence
        masked = []  # Accumulate characters/blank tokens for cloze text
        i = 0  # Character pointer within the sentence
        lower_s = s.lower()  # Lowercased sentence for matching
        while i < len(s):  # Walk across the sentence
            if lower_s[i:i + len(pattern)] == pattern:  # Detect keyword occurrence
                masked.append("____")  # Insert blank placeholder
                i += len(pattern)  # Jump past the keyword
            else:  # Otherwise, keep character
                masked.append(s[i])  # Copy original character
                i += 1  # Advance by one
        cloze = "".join(masked)  # Join parts into the cloze sentence
        cards.append((cloze, hit))  # Store the card and its answer
        if len(cards) >= max_cards:  # Respect maximum number of cards
            break  # Stop when enough cards are created
    return cards  # Return generated cards

# Streamlit
st.set_page_config(page_title= "Study Assistant", layout="wide", page_icon="📚")  # Configure page title and layout
st.title("Study Assistant")  # App title
st.subheader("Your AI powered study companion")  # Subtitle/description

# Global styles to polish visuals
st.markdown(
    """
    <style>
      .pdf-frame { border: 0; border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }
      .tag { display:inline-block; padding:6px 10px; margin:4px 6px 0 0; border-radius:16px; background:#eef2ff; color:#3730a3; font-size:0.9rem; }
      .stButton>button { border-radius:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar controls
st.sidebar.header("Controls")  # Sidebar heading
target_len = st.sidebar.slider("Summary length", 80, 250, 130, step=10)  # Model max length
top_k = st.sidebar.slider("Top keywords", 5, 20, 10)  # Number of keywords
max_cards = st.sidebar.slider("Max flashcards", 4, 20, 8)  # Number of cloze cards

 # Upload file
uploaded_file = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt"])  # File input widget
file_text = ""  # Accumulates extracted text from uploaded file

if uploaded_file is not None:  # Proceed only when a file is uploaded
    st.success(f"Uploaded: {uploaded_file.name}")  # Show upload confirmation
    file_text = ""  # Reset buffer for extracted text

    if uploaded_file.type == "application/pdf":  # PDF handling branch
        uploaded_file.seek(0)  # Rewind file pointer
        pdf_b64 = base64.b64encode(uploaded_file.read()).decode('utf-8')  # Base64-encode file for inline display

        uploaded_file.seek(0)  # Rewind again to extract text
        reader = PyPDF2.PdfReader(uploaded_file)  # Create PDF reader
        for p in reader.pages:  # Iterate through pages
            file_text += p.extract_text() or ""  # Append extracted text if present

    elif uploaded_file.type == "text/plain":  # Plain text handling branch
        file_text = uploaded_file.read().decode("utf-8")  # Read entire text file
    
    file_text = re.sub(r"\n+", " ", file_text)  # Replace runs of newlines with single spaces
    file_text = re.sub(r"\s{2,}", " ", file_text).strip()  # Collapse repeated whitespace and trim ends

    # Layout tabs for a cleaner UI
    tab_preview, tab_summary, tab_keywords, tab_cards = st.tabs(["Preview", "Summary", "Keywords", "Flashcards"])  # Main sections

    # Preview tab: show PDF or text preview
    with tab_preview:
        if uploaded_file.type == "application/pdf":
            st.markdown("### Preview")
            st.markdown(
                f'<iframe class="pdf-frame" src="data:application/pdf;base64,{pdf_b64}" width="100%" height="600px"></iframe>',
                unsafe_allow_html=True,
            )
        elif uploaded_file.type == "text/plain":
            st.markdown("### Preview")
            st.text_area("Text", file_text[:4000], height=350)

    # Summary tab: run model and display output
    with tab_summary:
        if st.button("Summarize text"):  # Button to trigger summarization
            with st.spinner("Summarizing..."):  # Visual feedback while summarizing
                summarizer = get_summarizer()  # Cached summarizer pipeline
                chunk = file_text[:2000]  # Use a manageable chunk size for the model
                min_len = max(30, int(target_len * 0.5))  # Derive a sane minimum length
                result = summarizer(chunk, max_length=target_len, min_length=min_len, do_sample=False)  # Run model
                summary = result[0]["summary_text"]  # Extract summary string

                st.subheader("Summary")  # Summary section heading
                st.write(summary)  # Display the generated summary
                st.success("Summarization complete!")  # Success notification

    # Keywords tab: compute and show as rounded tags
    keywords = extract_keywords(file_text, max_k=top_k) if file_text else []
    with tab_keywords:
        st.markdown("### Keywords")
        if keywords:
            st.markdown(" ".join([f"<span class='tag'>{k}</span>" for k in keywords]), unsafe_allow_html=True)
        else:
            st.info("No keywords found.")

    # Flashcards tab: generate and show cloze cards
    with tab_cards:
        if keywords:
            if st.button("Generate flashcards"):
                cards = make_cloze_cards(file_text, keywords, max_cards=max_cards)
                if cards:
                    st.markdown("### Cloze Flashcards")
                    for idx, (cloze, answer) in enumerate(cards, start=1):
                        st.write(f"{idx}. {cloze}")
                        with st.expander("Show answer"):
                            st.write(answer)
                else:
                    st.info("No suitable sentences found for flashcards.")
        else:
            st.info("No keywords available — upload text and try again.")
    
