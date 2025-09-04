from langchain_ollama import ChatOllama

llm = ChatOllama(model="deepseek-r1:1.5b", validate_model_on_init=True)

content = llm.invoke("The first man on the moon was ...").content
print(content)

# pip install -qU langchain_ollama
# ollama pull deepseek-r1:1.5b