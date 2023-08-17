import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.llms import HuggingFaceHub, LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain

# Step 1: Load Data
pdf_path = "code_of_conduct.pdf"
loader = PyPDFLoader(pdf_path)
data = loader.load()

# Step 2: Split Text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(data)

# Step 3: Load Embeddings
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Step 4: Initialize Pinecone
index_name = "langchain"
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# Step 5: Create Embeddings
docsearch = Pinecone.from_texts([t.page_content for t in docs], embeddings, index_name=index_name)

# Streamlit App
def main():
    st.title("University Rules Knowledgebase")

    # Sidebar
    st.sidebar.title("Ask a Question")
    query = st.sidebar.text_input("Enter your question")

    if st.sidebar.button("Search"):
        search_results = docsearch.similarity_search(query)
        show_results(search_results, query)

def show_results(results, query):
    st.subheader("Search Results")
    
    for doc in results:
        st.write(f"Document #{doc.id}:")
        st.write(doc.text)

        # Initialize LLMS models
        llm_hf = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
        llm_llama = LlamaCpp(
            model_path=model_path,
            max_tokens=256,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            callback_manager=callback_manager,
            n_ctx=1024,
            verbose=False,
        )

        # Load QA chains
        chain_hf = load_qa_chain(llm_hf, chain_type="stuff")
        chain_llama = load_qa_chain(llm_llama, chain_type="stuff")

        # Run chains
        st.subheader("Answer from Hugging Face Model:")
        answer_hf = chain_hf.run(input_documents=[doc.text], question=query)
        st.write(answer_hf)

        st.subheader("Answer from Llama Model:")
        answer_llama = chain_llama.run(input_documents=[doc.text], question=query)
        st.write(answer_llama)

# Run the Streamlit app
if __name__ == "__main__":
    main()
