# import os 
# from dotenv import load_dotenv
# from openai import OpenAI 

# load_dotenv()


# client = OpenAI(
#     api_key = os.getenv("OPENAI_API_KEY")
# )


# text = "python is a programming language used for AI and web development."

# sentences = [
#              "Python is a programming Language.",
#              "Python is commonly used for artificial intelligence.",
#              "The Capital of India is New Delhi.",
#              "Dogs are popular domestic animals."
#             ]

# response = client.embeddings.create(
#     model = "text-embedding-3-small",
#     input = sentences
# )

# # embedding = response.data[0].embedding

# # print("Dimensions:", len(embedding))
# # print("First 10 values:", embedding[:10])

# # for i,item in enumerate(response.data):
# #     print(f"Sentence {i+1}")
# #     print("Embedding dimensions:", len(item.embedding))
# #     print("Embedding dimensions:", item.embedding[:10])
# #     print()


# question = "What programming language is used for AI?"


import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key= os.getenv("OPENAI_API_KEY")
)

def cosine_similarity(a,b):
    va = np.array(a)
    vb = np.array(b)

    return np.dot(va,vb)/(np.linalg.norm(va)*np.linalg.norm(vb))


documents = [
    "Python is a programming language.",
    "Python is widely used in AI.",
    "Current temp in hyd is 12 degrees.",
    "Dogs are popular domestic animals.",
    "OpenAI provides APIs for building AI applications."
]

query = "What does OpenAI provide?"

query_response   = client.embeddings.create(
    model = "text-embedding-3-small",
    input = query 
)

query_embedding =  query_response.data[0].embedding

document_response = client.embeddings.create(
    model = "text-embedding-3-small",
    input = documents
)

document_embeddings = [item.embedding for item in document_response.data]

scores = []

for document,embedding in zip(documents,document_embeddings):
    score = cosine_similarity(query_embedding,embedding)
  
    scores.append([score,document])

scores.sort(key=lambda x: x[0],reverse=True)

top_results = scores[:2]

context = "\n".join(
    document for score, document in top_results
)



prompt = f"""
Answer the user's question using only the context provided below.

Context:
{context}

Question:
{query}

If the answer cannot be found in the context, say:
"I don't have enough information to answer this."
"""

response = client.responses.create(
    model="gpt-5-mini",
    input=prompt
)

print(response.output_text)





