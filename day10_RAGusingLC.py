##########  VECTOR STORE #############

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


## load env variables

load_dotenv()

## load the document

loader = TextLoader("sample.txt")
documents = loader.load()

##Split it into Chunks

splitter = RecursiveCharacterTextSplitter(
    chunk_size=70,
    chunk_overlap=10
)

chunks = splitter.split_documents(documents)

# print(chunks)


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma.from_documents(
    documents = chunks,
    embedding=embeddings,
    collection_name = "employee_policy"
)

retriever = vector_store.as_retriever(
    search_kwargs = {"k":2}
)

#Ask a Question

results = retriever.invoke("Under how many days travel realted costs must be submitted?")

# for i,document in enumerate(results):
#     print(f"Result {i+1}:")
#     print(document.page_content)

model = ChatOpenAI(model = "gpt-5-mini")


prompt = ChatPromptTemplate.from_template("""
Answer the question only using the context provided.
if the answer is not in the context, say I don't know.

Context:
{context}

Question:
{question}
"""
)

def format_documents(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )



rag_chain = (
    {"context": retriever | format_documents,
     "question": RunnablePassthrough()

    }

    | prompt
    | model 
    | StrOutputParser()
        )

question = "How many leaves does employee gets in an year?"

answer = rag_chain.invoke(question)

print("Answer:",answer)