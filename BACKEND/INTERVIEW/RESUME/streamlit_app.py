import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import streamlit as st
from BACKEND.INTERVIEW.Utils.util import extract_text_and_links_from_pdf
from BACKEND.INTERVIEW.RESUME.graph import build_graph

def main():
    st.title("📄 Resume Uploader")

    uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])

    if uploaded_file is not None:
        st.success("✅ Resume uploaded successfully!")
        st.write(f"**Filename**: {uploaded_file.name}")

        # with st.expander("Preview File Info"):
            # st.write(f"Filename: {uploaded_file.name}")
            # st.write(f"File Type: {uploaded_file.type}")
            # st.write(f"File Size: {uploaded_file.size / 1024:.2f} KB")

        if st.button("📤 Send to Backend for Parsing"):
            st.info("Sending resume to backend for parsing... 🚀")
            with st.spinner("Extracting data..."):
                # 1. Extract text and links
                text, links = extract_text_and_links_from_pdf(uploaded_file)
                st.success("Text and links extracted!")

                # st.subheader("🔍 Extracted Text")
                # st.text_area("Text", text, height=300)

                # st.subheader("🔗 Extracted Links")
                # for link in links:
                #     st.write(link)

            with st.spinner("🔧 Running Resume Graph..."):
                # 2. Build the graph
                graph = build_graph()

                # 3. Prepare initial state
                input_state = {
                    "full_text": text,
                    "links": links
                }

                # 4. Run the graph with state
                final_state = graph.invoke(input_state)

                # 5. Display final output state
                st.success("🎯 Resume data parsed successfully!")
                st.subheader("📊 Final Output")
                st.write(final_state["message"])

if __name__ == "__main__":
    main()