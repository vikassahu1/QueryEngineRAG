import streamlit as st
import os
from QAWithPDF.data_ingestion import load_data
from QAWithPDF.embedding import download_gemini_embedding
from QAWithPDF.model_api import load_model

def main():
    st.set_page_config("Info Retrival System", layout="wide")
    
    # Define the directory to save uploaded files
    upload_dir = "Data"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # Clear the Data directory before saving new files
    for filename in os.listdir(upload_dir):
        file_path = os.path.join(upload_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    st.title("📄Information Retrieval System")
    st.subheader("Upload your documents and ask questions based on their content.")

    # File uploader for multiple files
    docs = st.file_uploader("Upload your documents", type=["pdf", "docx"], accept_multiple_files=True)

    user_question = st.text_input("💬 Ask your question")

    if st.button("Submit & Process"):
        if docs:
            for doc in docs:
                # Save the uploaded files
                file_path = os.path.join(upload_dir, doc.name)
                with open(file_path, "wb") as f:
                    f.write(doc.getbuffer())

            with st.spinner("Processing..."):
                documents = load_data(upload_dir)  # Load the documents from the specified directory
                model = load_model()
                query_engine = download_gemini_embedding("AIzaSyBHrK333hcGb6lERHDm3aieshj_9oBVJJM", documents)
                
                response = query_engine.query(user_question)
                
                # st.success("✅ Query processed successfully!")
                st.write("Response:")
                st.write(response.response)
        else:
            st.warning("Please upload at least one document.")

    # Footer section for additional info
    st.markdown("---")
    st.subheader("📚 About This App")
    st.write("""
        This application allows you to upload documents and ask questions about their content. 
        It uses advanced information retrieval techniques to provide accurate answers based on the uploaded documents.
    """)

if __name__ == "__main__":
    main()
