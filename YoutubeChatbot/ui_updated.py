import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import time

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="YouTube ChatBot",
    page_icon="🤖",
    layout="centered"
)

# Title and description
st.title("🎬 YouTube ChatBot")
st.caption("Ask questions about any YouTube video with subtitles")

st.divider()

# Initialize session state
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# Video Loading Section
st.subheader("📹 Load Video")

col1, col2 = st.columns([3, 1])

with col1:
    video_id = st.text_input(
        "Video ID",
        placeholder="Enter YouTube video ID (e.g., dQw4w9WgXcQ)",
        label_visibility="collapsed"
    )

with col2:
    submit_video = st.button("Load", use_container_width=True, type="primary")

if submit_video and video_id:
    with st.spinner("Loading transcript..."):
        try:
            loader = YoutubeLoader(video_id=video_id)
            st.session_state.transcript = loader.load()
            
            # Pre-process vectorstore
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(st.session_state.transcript)
            embeddings = OpenAIEmbeddings()
            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
            
            st.success("✅ Transcript loaded successfully!")
            
        except Exception as e:
            st.error("❌ Error: Please check if the video has subtitles enabled")
            st.session_state.transcript = None
            st.session_state.vectorstore = None

# Question Section
if st.session_state.transcript is not None:
    st.divider()
    st.subheader("💬 Ask Questions")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        question = st.text_input(
            "Question",
            placeholder="What would you like to know about this video?",
            label_visibility="collapsed"
        )
    
    with col2:
        submit_question = st.button("Ask", use_container_width=True, type="primary")
    
    if submit_question and question:
        with st.spinner("Thinking..."):
            try:
                # Chains Logic
                parser = StrOutputParser()
                
                def format_text(context):
                    context_text = "\n\n".join(text.page_content for text in context)
                    return context_text
                
                retriever = st.session_state.vectorstore.as_retriever()
                
                parallel_chain = RunnableParallel({
                    'context': retriever | RunnableLambda(format_text),
                    'question': RunnablePassthrough()
                })
                
                prompt = PromptTemplate(template='''
                    You are a helpful assistant.
                    Your job is to answer the questions asked based on the following context only:
                    
                    Context: {context}
                    
                    Question: {question}
                    
                    Provide a clear, concise answer based only on the context provided.
                ''')
                
                model = ChatOpenAI()
                seq_chain = prompt | model | parser
                final_chain = parallel_chain | seq_chain
                
                result = final_chain.invoke(question)
                
                st.divider()
                st.subheader("💡 Answer")
                
                def streamer():
                    for text in result:
                        time.sleep(0.02)
                        yield text
                
                st.write_stream(streamer)
                
            except Exception as e:
                st.error(f"❌ Error processing question: {str(e)}")
    
    elif submit_question and not question:
        st.warning("⚠️ Please enter a question")

# Footer
st.divider()
st.caption("💡 Tip: Ask specific questions for better results")