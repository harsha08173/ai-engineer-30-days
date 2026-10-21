from langchain_core.runnables import RunnableLambda,RunnablePassthrough,RunnableParallel

def make_uppercase(text):
    return text.upper()

def get_length(text):
    return len(text)


chain = RunnableParallel(original=RunnablePassthrough(),upper=RunnableLambda(make_uppercase),length=RunnableLambda(get_length))


response = chain.invoke("Harsha")
print(response)
print(response["original"])
print(response["upper"])
print(response["length"])




# chain  = RunnableLambda(make_uppercase)
# chain1 = RunnablePassthrough()



# response = chain.invoke("hello harsha")
# print(response)

# response1 = chain1.invoke("hello harsha")
# print(response1)