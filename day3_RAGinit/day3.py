import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()

client  = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)



with open("company_policy.txt","r") as file:
    text = file.read()

chunk_size = 200

chunks = []

# for i in range(0,len(text),chunk_size):
#     chunk = text[i:i+chunk_size]
#     chunks.append(chunk)

# for i,chunk in enumerate(chunks):
#     print(f"\n--Chunk {i+1} --")
#     print(chunk)

chunks = text.split("\n\n")

response = client.embeddings.create(
    model = "text-embedding-3-small",
    input = chunks
)

embeddings = [item.embedding for item in response.data]

print("Number of chunks:", len(chunks))
print("Number of embeddings:",len(embeddings))
print("Embedding dimesions:",len(embeddings[0]))


chroma_client = chromadb.Client()
collection = chroma_client.create_collection(
    name="company_policy"
)


collection.add(
    ids = [str(i) for i in range(len(chunks))],
    documents=chunks,
    embeddings = embeddings,
    metadatas = [
                {"department":"HR",
                  "document_type":"company_policy"
                } for _ in chunks
                ]

)




# results = collection.get(include=[embeddings])

# print(results)



def askQuestion(question):
    # query = "What is the company's maternity leave policy?"
    query = question

    query_response = client.embeddings.create(
        model = "text-embedding-3-small",
        input = query
    )

    query_embedding = query_response.data[0].embedding

    # results = collection.query(
    #     query_embeddings = [query_embedding],
    #     n_results = 3
    # )

    results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    where={
        "department": "HR"
    }
)

    # print(results["documents"])

    retrieved_documents = results["documents"][0]

    context = "\n\n".join(retrieved_documents)

    # print("Context - ")
    # print(context)
    # print("\n\n")


    prompt = f"""
    You are a helpful assistant.

    Answer the question using ONLY the context provided below.

    If the answer cannot be found in the context, say:
    "I don't have enough information to answer this."

    Do not use outside knowledge.

    Context:
    {context}

    Question:
    {query}
    """

    response = client.responses.create(
        model = "gpt-5-mini",
        input = prompt
    )

    print(response.output_text)

# askQuestion("What is the maternity leave policy?")

# askQuestion("How many leave days can employees carry forward?")

# askQuestion("What is the company's travel reimbursement policy?")

# askQuestion("Who founded the company?")