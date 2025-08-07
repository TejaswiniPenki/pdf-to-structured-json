import streamlit as st
import pdfplumber
import fitz  # PyMuPDF
import google.generativeai as genai
import pandas as pd
import os

# ---- Gemini Setup ----
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # OR hardcode key for local dev
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")  # Fast & reliable

# ---- Helper: Extract text from PDF ----
def extract_text_from_pdf(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = "\n".join([page.get_text() for page in doc])
    return text

# ---- Helper: Extract tables from PDF ----
def extract_tables_from_pdf(uploaded_file):
    tables = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_tables()
            for table in extracted:
                df = pd.DataFrame(table[1:], columns=table[0])
                tables.append(df)
    return tables

# ---- Gemini Prompt ----
def generate_structured_output(text, tables):
    prompt = f"""
You are an expert in document analysis.

Given the following PDF content (text and tables), extract and organize it into a clean JSON format.

From text:
{{
  "Document Title": "<title>",
  "Sections": [
    {{
      "Heading": "<heading>",
      "Content": "<text>"
    }},
    {{
      "Heading": "<heading>",
      "Subsections": [
        {{
          "Subheading": "<subheading>",
          "Content": "<text>"
        }}
      ]
    }}
  ]
}}

From tables:
[
  {{
    "Field": "...",
    "Value": "..."
  }},
  ...
]

Text Content:
{text}

Table Data:
{[df.to_dict(orient='records') for df in tables]}
"""
    response = model.generate_content(prompt)
    return response.text

# ---- Streamlit App UI ----
st.set_page_config(page_title="PDF Data Extractor", layout="wide")
st.title("📄 AI-Powered PDF Extractor")
st.markdown("Upload a PDF with text, tables, or descriptions. This app extracts & structures content using Google Gemini.")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    st.success("✅ PDF uploaded successfully!")

    with st.spinner("🔍 Extracting content from PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        tables = extract_tables_from_pdf(uploaded_file)

    st.subheader("📘 Extracted Text")
    st.text_area("Text Content", text, height=250)

    st.subheader("📊 Extracted Tables")
    if tables:
        for i, df in enumerate(tables):
            st.markdown(f"**Table {i+1}:**")
            st.dataframe(df)
    else:
        st.info("No tables found in the document.")

    if st.button("🧠 Generate Structured JSON"):
        with st.spinner("🛠 Generating with Gemini..."):
            result = generate_structured_output(text, tables)
        st.subheader("🧾 Final Structured Output")
        st.code(result, language="json")
        st.download_button("📥 Download JSON", result, file_name="structured_output.json")
else:
    st.info("Please upload a PDF file to proceed.")
