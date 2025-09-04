import getpass
import os
from langchain.chat_models import init_chat_model

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

model = init_chat_model("gpt-4o-mini", model_provider="openai")
model.invoke("Hello, world!")

# # Returns a langchain_openai.ChatOpenAI instance.
# gpt_4o = init_chat_model("gpt-4o", model_provider="openai", temperature=0)
# # Returns a langchain_anthropic.ChatAnthropic instance.
# claude_opus = init_chat_model(
#     "claude-3-opus-20240229", model_provider="anthropic", temperature=0
# )
# # Returns a langchain_google_vertexai.ChatVertexAI instance.
# gemini_15 = init_chat_model(
#     "gemini-2.5-pro", model_provider="google_genai", temperature=0
# )
#
# # Since all model integrations implement the ChatModel interface, you can use them in the same way.
# print("GPT-4o: " + gpt_4o.invoke("what's your name").content + "\n")
# print("Claude Opus: " + claude_opus.invoke("what's your name").content + "\n")
# print("Gemini 2.5: " + gemini_15.invoke("what's your name").content + "\n")