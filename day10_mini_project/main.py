from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain.tools import tool
from langchain.agents import create_agent


load_dotenv()                              #loads the env. variables to this scope

loader = TextLoader("python_notes.txt")    ##Reads the text file
documents = loader.load()                  ##Returns the list of Langchain documents objects

# print(documents[0].page_content)         ##Print actual text inside first document

splitter = RecursiveCharacterTextSplitter(
           chunk_size = 150,   #targets chunks around 150 characters
           chunk_overlap = 20  #tries to preserve about 20 characters between adjacent chunks.
            )

chunks = splitter.split_documents(documents) #Split the documents while preserving the meta data
# print("Total chunks:",len(chunks))

# for i,chunk in enumerate(chunks):
#     print(f"\nChunk {i+1}")
#     print(chunk.page_content)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small") #OpenAIEmbeddings converts each chunk into a numerical vector.

vector_store = Chroma.from_documents( #Chroma.from_documents() stores the chunks and their embeddings for similarity search.
    documents = chunks,  
    embedding=embeddings,
    collection_name = "Python_documentation"
)

# print("Stored chunks:",vector_store._collection.count())
 
retriever = vector_store.as_retriever(
    search_kwargs = {"k":2} #retriving the top 2 similar results from the vector db
)

results = retriever.invoke("What is a lamda function?")

# for doc in results:  #printing the fetched results
#     print(doc.page_content)

model = ChatOpenAI(model="gpt-5-mini")
prompt = ChatPromptTemplate.from_template("""

Answer the question the using only the context below.
{context}

Question:
{question}

""")

def format_documents(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_documents,
     "question": RunnablePassthrough()
    } 
    | prompt
    | model
    | StrOutputParser()
)

answer = rag_chain.invoke("What is a lambda function.")
print("\nAnswer:",answer)

@tool
def calculator(a: float, b: float, operation: str) -> float:
    """Perform addition, subtraction, multiplication, or division on two numbers."""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            return "Cannot divide by zero"
        return a / b
    else:
        return "Unsupported operation"


@tool
def search_python_docs(question:str)->str:
    """Search python documentation notes to answer questions about python concepts"""
    docs = retriever.invoke(question)
    return "\n\n".join(doc.page_content for doc in docs)


agent = create_agent(
    model = model,
    tools = [calculator,search_python_docs],
    system_prompt = ("You are a Developer Documentation Assistant."
        "Use search_python_docs for questions about Python concepts."
        "Use calculator for arithmetic questions."
        "Answer clearly and do not invent facts from the documentation.")
)

# response = agent.invoke(
#     {
#         "messages":[
#             {"role":"user","content":"What is the lambda function in python. in one sentence?"}
#         ]
#     }
# )
# print(response["messages"][-1].content)


questions = [
    "What is a lambda function in Python in 6 words?",
    "What is 125 multiplied by 8?"
]

# for question in questions:
#     response = agent.invoke({
#         "messages": [
#             {"role": "user", "content": question}
#         ]
#     })

#     print(f"\nQuestion: {question}")
#     print("Answer:", response["messages"][-1].content)

#     for message in response["messages"]:
#         print(type(message).__name__, message.content)

conversation = []

while(True):
    question = input("\You: ")

    if question.lower()=="exit":
        break

    conversation.append({"role":"user","content":question})
    response = agent.invoke({"messages":conversation})
    conversation = response["messages"]

    print("Assistant:",conversation[-1].content)
