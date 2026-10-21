from langchain_core.documents import Document 
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv 
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder



load_dotenv()


policy_text = """
Annual leave policy: Employees recieve 20 days of annual leave.
Carry forward policy: Employees can carry forward up to 5 days of unused leave.
Password policy: Employees must change their password every 90 days.
VPN policy: Employes must use VPN for secure remote access.
Travel policy: Employees must submit travel expenses with in 30 days.
Hotel expenses: Hotel expenses can be claimed during the approved business travel.
"""


splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20 
)   

documents = splitter.create_documents([policy_text],metadatas=[{"department":"HR","document_type":"policy"}])

# print("Total chunks:",len(documents))

# for doc in documents:
#     print(doc.page_content)
#     print("-----")

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3

embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-small"
)

vector_store = Chroma.from_documents(
    documents = documents,
    embedding = embeddings,
    collection_name="day14_hybrid_search_metadata"
)

vector_retriever = vector_store.as_retriever(
    search_kwargs = {"k":3,"filter":{"department":"HR"}},
)

query = "How many leave days can employee carry forward?"


print("\n--- BM25 Results ---")
for doc in bm25_retriever.invoke(query):
    print(doc.page_content)

print("\n---Vector Results ---")
for doc in vector_retriever.invoke(query):
    print(doc.page_content)

hybrid_retriever = EnsembleRetriever(
        retrievers = [bm25_retriever,vector_retriever],
        weights = [0.5,0.5]
    )

query = "Under how many days employees must submit travel expenses?"

results = hybrid_retriever.invoke(query)


print("\n--- Hybrid Results ---")
for doc in results:
    print(doc.page_content)
    print("-----")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

pairs = [
    (query, doc.page_content)
    for doc in results
]

scores = reranker.predict(pairs)

reranked_results = sorted(
    zip(results, scores),
    key=lambda item: item[1],
    reverse=True
)

print("\n--- Reranked Results ---")
for doc, score in reranked_results:
    print(f"Score: {score:.4f}")
    print(doc.page_content)
    print("-----")