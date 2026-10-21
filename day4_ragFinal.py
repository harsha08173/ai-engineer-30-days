#Dynamic tool calling in multiple tools 


import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb


load_dotenv()

client  = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)


chunks = [
    "Employees may carry forward up to 5 unused leave days to the following year.",
    "Employees receive 20 days of annual leave every year.",
    "Employees can request leave through the internal leave management system.",
    "Employees must change their password every 90 days.",
    "The IT department provides VPN access for remote employees.",
    "Travel expenses must be submitted within 30 days.",
    "Employees can claim hotel expenses during approved business travel."
]

metadatas = [
    {"department": "HR", "document_type": "leave_policy"},
    {"department": "HR", "document_type": "leave_policy"},
    {"department": "HR", "document_type": "leave_policy"},
    {"department": "IT", "document_type": "security_policy"},
    {"department": "IT", "document_type": "remote_work_policy"},
    {"department": "Finance", "document_type": "travel_policy"},
    {"department": "Finance", "document_type": "travel_policy"}
]


response = client.embeddings.create(
    model = "text-embedding-3-small",
    input = chunks
)

embeddings = [item.embedding for item in response.data]

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(
    name="company_policy"
)


collection.add(
    ids = [str(i) for i in range(len(chunks))],
    documents=chunks,
    embeddings = embeddings,
    metadatas = metadatas

)

query = input("What is your query?")

query_response = client.embeddings.create(
    model = "text-embedding-3-small",
    input = query
)

query_embedding = query_response.data[0].embedding

results = collection.query(
query_embeddings=[query_embedding],
n_results=6,

)

retrieved_documents = results["documents"][0]

print("** retrieved_documents **")
for i in retrieved_documents:
    print(i)

def rerank_documents(query,documents):
    candidates = ""
    for i,document in enumerate(documents):
        candidates += f"""
        candidate {i+1}:
        {document}
    """
   

    rerank_prompt = f"""
    You are a document relevance evaluator.

    Given the user's question and candidate documents,
    rank the documents from most relevant to least relevant.

    User question:
    {
        query
    }

    candidate documents:

    {
        candidates
    }
    
    return only the candidate numbers in order.

    Example:
    
    2,1,3 """

    response = client.responses.create(
        model="gpt-5-mini",
        input= rerank_prompt
        )
    ranking = response.output_text

    return ranking



sequence = rerank_documents(query,retrieved_documents)


final_context = ""




for i in range(0,3,2):
    final_context = final_context+retrieved_documents[int(sequence[i])-1]+"\n"

print("** final context **")
print(final_context)

prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the context provided below.

If the answer cannot be found in the context, say:
"I don't have enough information to answer this."

Do not use outside knowledge.

Context:
{final_context}

Question:
{query}
"""

response = client.responses.create(
    model = "gpt-5-mini",
    input = prompt
)

print("** Response **")
print(response.output_text)