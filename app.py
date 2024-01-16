__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import pandas as pd
from dotenv import load_dotenv
import fitz  # install using: pip install PyMuPDF
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain import PromptTemplate

os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_KEY']

# side bar contents
with st.sidebar:
    st.title('🤗💬 Smart Search')
    st.markdown("""
    ## Краткая информация
    
    - [Streamlit](https://streamlit.io/)
    - [Langchain](https://python.langchian.com/)
    - Локальный ИИ на базе LLama70
                
    """)
    add_vertical_space(1)
    st.write("Made by MPK")

load_dotenv()


def main():
    st.header("Smart Search 💬")
    tab1, tab2 = st.tabs(["Умное сравнение", "Загрузить новые знания"])

    # upload a PDF file
    with tab1:
        st.subheader("Загрузите PDF файл, чтобы сравнить его и найти противоречия")
        pdf = st.file_uploader("Загрузите PDF файл", type="pdf")
        if pdf is not None:
            vectordb = Chroma(persist_directory="./data", embedding_function=OpenAIEmbeddings())

            temp_file = "./temp_1.pdf"
            with open(temp_file, "wb") as file:
                file.write(pdf.getvalue())
                file_name = pdf.name
            with fitz.open(temp_file) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0, keep_separator=False, separators=['.','\n'] )
            documents = text_splitter.split_text(text)
            template = """
             Your task is to compare these texts assess compliance. 
             When you find a discrepancy or contradictions, you respond only "1", else "0
            Text 1: " {context} "

            Text 2: " {question} "
            Answer: 
            """
            PROMPT = PromptTemplate(template=template, input_variables=['question', 'context'])
            qa_chain = RetrievalQA.from_chain_type(
                llm=OpenAI(model_name="gpt-4-1106-preview", max_tokens=1000, temperature=0, top_p=1.0),
                retriever=vectordb.as_retriever(search_kwargs={"k":1}),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True,
                verbose=True
            )

            texts = text.split('.')
            columns = ['Text', 'Source Documents']
            df = pd.DataFrame(columns=columns)
            for t in texts:
                if len(t) > 20:
                    # we can now execute queries against our Q&A chain
                    print(qa_chain.get_prompts(PROMPT))
                    result = qa_chain({'query': t})
                    if result['result'] == '1':
                        if len(result['source_documents']) > 0:
                            source = str(result['source_documents'][0].page_content)
                        else:
                            source = ""
                        df = pd.concat([df, pd.DataFrame([
                            [result['query'], source]], columns=columns)
                                        ], ignore_index=True)
            if len(df) > 0:
                st.header('Найденные противоречия', divider='violet')
                st.dataframe(df)
            else:
                st.write("Противоречий не найдено")

    with tab2:
        new_pdf = st.file_uploader("Update your knowledge", type="pdf")
        if new_pdf is not None:
            data_load_state = st.text('Загрузка...')

            temp_file = "./temp.pdf"
            with open(temp_file, "wb") as file:
                file.write(new_pdf.getvalue())
                file_name = new_pdf.name
            loader = PyPDFLoader(temp_file)
            documents = loader.load()

            # we split the data into chunks of 1,000 characters, with an overlap
            # of 200 characters between the chunks, which helps to give better results
            # and contain the context of the information between chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0, keep_separator=False, separators=['.'] )
            documents = text_splitter.split_documents(documents)
            print(documents)

            # we create our vectorDB, using the OpenAIEmbeddings tranformer to create
            # embeddings from our text chunks. We set all the db information to be stored
            # inside the ./data directory, so it doesn't clutter up our source files
            vectordb = Chroma.from_documents(
                documents,
                embedding=OpenAIEmbeddings(),
                persist_directory='./data'
            )
            vectordb.persist()
            data_load_state.text('Загрузка завершена!')
    # st.write(pdf)


if __name__ == "__main__":
    main()
