# from langchain_openai import ChatOpenAI
# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from dotenv import load_dotenv

# load_dotenv()


# loader = TextLoader("sample.txt")
# documents = loader.load()

# print("Number of documents", len(documents))
# print("Document content", documents[0].page_content)


# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=70,
#     chunk_overlap=10,
    
# )

# chunks = splitter.split_documents(documents)

# print("Number of chunks: ", len(chunks))
# print(chunks)
# for i, chunk in enumerate(chunks):
#     print(f"\nChunk {i+1}:")
#     print(chunk.page_content)


# model = ChatOpenAI(model="gpt-5-mini")

# # 1. Invoke: one input
# response = model.invoke("Explain RAG in one sentence.")
# print("INVOKE:", response.content)

# # 2. Batch: multiple inputs
# responses = model.batch([
#     "Explain embeddings in one sentence.",
#     "Explain LangChain in one sentence."
# ])

# for i,response in enumerate(responses):
#     print(f"BATCH RESPONSE {i+1}:", response.content)


# for chunck in model.stream("Explain AI agent in two sentences."):
#     print(chunck.content,end="",flush=True)
  
##########  VECTOR STORE #############

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


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

retriver = vector_store.as_retriever(
    search_kwargs = {"k":2}
)

#Ask a Question

results = retriver.invoke("Under how many days travel realted costs must be submitted?")

for i,document in enumerate(results):
    print(f"Result {i+1}:")
    print(document.page_content)