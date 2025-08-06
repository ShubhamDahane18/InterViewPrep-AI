import fitz  # PyMuPDF

def extract_text_and_links_from_pdf(pdf):
    full_text = ""
    links = []

    pdf = fitz.open(stream=pdf.read(), filetype="pdf")
    
    for page_num, page in enumerate(pdf, start=1):
        full_text += page.get_text()

        # Extract links on this page
        for link in page.get_links():
            uri = link.get("uri", None)
            rect = link.get("from", None)
            if uri and rect:
                # Get anchor text behind the link
                anchor_text = page.get_textbox(rect)
                links.append({
                    "page": page_num,
                    "text": anchor_text.strip(),
                    "url": uri.strip()
                })

    pdf.close()

    return full_text, links


from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro" , api_key=os.getenv("GOOGLE_API_KEY"))

# llm = ChatGroq(model="llama-3.3-70b-versatile")

def load_llm():
    return llm

