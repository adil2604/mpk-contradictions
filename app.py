__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import pandas as pd
from dotenv import load_dotenv
import fitz  # install using: pip install PyMuPDF
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain import PromptTemplate

os.environ["OPENAI_API_KEY"] = "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"

# side bar contents
with st.sidebar:
    st.title('🤗💬 Contradicton Finder by AI')
    st.markdown("""
    ## About
    This app is an LLM-powered finder built using:
    - [Streamlit](https://streamlit.io/)
    - [Langchain](https://python.langchian.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
                
    """)
    add_vertical_space(3)
    st.write("Made by MPK")

load_dotenv()


def main():
    st.header("Compare your documents 💬")
    tab1, tab2 = st.tabs(["Compare", "Update Knowledge"])

    # upload a PDF file
    with tab1:
        st.subheader("Load your document to compare with knowledge")
        pdf = st.file_uploader("Upload your PDF", type="pdf")
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
            template = """
            Сompare two texts, if the texts meaning do not contradict each other print only "0", otherwise print "1"
            Text 1: {context}

            Text 2: {question}
            Answer: 
            """
            PROMPT = PromptTemplate(template=template, input_variables=['question', 'context'])
            qa_chain = RetrievalQA.from_chain_type(
                llm=OpenAI(),
                retriever=vectordb.as_retriever(search_kwargs={'k': 1}),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )

            texts = text.split('.')
            columns = ['Text', 'Source Documents']
            df = pd.DataFrame(columns=columns)
            for t in texts:
                if len(t) > 10:
                    # we can now execute queries against our Q&A chain
                    result = qa_chain({'query': t})
                    if result['result'] == '1':
                        if len(result['source_documents']) > 0:
                            source = result['source_documents'][0].page_content
                        else:
                            source = ""
                        df = pd.concat([df, pd.DataFrame([
                            [result['query'], source]], columns=columns)
                                        ], ignore_index=True)
            if len(df) > 0:
                st.header('Найденные не состыковки', divider='violet')
                st.dataframe(df)
            else:
                st.write("No contradictions found")

    with tab2:
        new_pdf = st.file_uploader("Update your knowledge", type="pdf")
        if new_pdf is not None:
            data_load_state = st.text('Loading data...')

            temp_file = "./temp.pdf"
            with open(temp_file, "wb") as file:
                file.write(new_pdf.getvalue())
                file_name = new_pdf.name
            loader = PyPDFLoader(temp_file)
            documents = loader.load()

            # we split the data into chunks of 1,000 characters, with an overlap
            # of 200 characters between the chunks, which helps to give better results
            # and contain the context of the information between chunks
            text_splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=50)
            documents = text_splitter.split_documents(documents)

            # we create our vectorDB, using the OpenAIEmbeddings tranformer to create
            # embeddings from our text chunks. We set all the db information to be stored
            # inside the ./data directory, so it doesn't clutter up our source files
            vectordb = Chroma.from_documents(
                documents,
                embedding=OpenAIEmbeddings(),
                persist_directory='./data'
            )
            vectordb.persist()
            data_load_state.text('Loading data...done!')
    # st.write(pdf)





if __name__ == "__main__":
    main()
