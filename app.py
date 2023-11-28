import streamlit as st
from script import get_pdf_text, get_text_chunks, get_vectorstore, retrieval_qa_chain

@st.cache_data
def load_data(path_to_pdf):
    raw_text = get_pdf_text([path_to_pdf])
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    db = vectorstore.as_retriever(search_kwargs={'k': 3})
    return db

path_to_pdf = './addicted to silence final V1.pdf'
db = load_data(path_to_pdf)
bot = retrieval_qa_chain(db, True)

@st.cache_data
def get_llm_response(query):
    matching_docs = db.similarity_search(query)
    answer = bot.run(input_documents=matching_docs, question=query)
    return answer

st.set_page_config(page_title="Doc Searcher", page_icon=":robot_face:")
st.header("Query PDF Source")
form_input = st.text_input('Enter Query')
submit = st.button("Generate")

if submit:
    response = get_llm_response(form_input)
    st.write("Response:", response['result'])
    st.write("Source Documents:", response['source_documents'])
