test_questions = [
    {
        "question": "How often must employees change their passwords?",
        "expected_answer": "90 days",
        "should_answer": True,
    },
    {
        "question": "How many days of annual leave do employees receive?",
        "expected_answer": "20 days",
        "should_answer": True,
    },
    {
        "question": "How many days of annual leave can employees carry over?",
        "expected_answer": "5 days",
        "should_answer": True,
    },
    {
        "question": "Within how many days must travel expenses be submitted?",
        "expected_answer": "30 days",
        "should_answer": True,
    },
    {
        "question": "What is the company's work-from-Mars policy?",
        "expected_answer": None,
        "should_answer": False,
    },
    {
        "question": "Who won the FIFA World Cup in 1998?",
        "expected_answer": None,
        "should_answer": False,
    },
]


from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI 
from langchain_chroma import Chroma 
from langchain_core.prompts import ChatPromptTemplate 
