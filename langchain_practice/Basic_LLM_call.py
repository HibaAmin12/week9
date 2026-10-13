# ---------- Load secrets ----------
from dotenv import load_dotenv
load_dotenv()  # Reads OPENAI_API_KEY from the .env file into the environment

# ---------- Imports ----------
from langchain_openai import ChatOpenAI   # LangChain wrapper around OpenAI chat models

# ---------- Create the model ----------
# gpt-4o-mini: cheap and fast model
# temperature=0: least randomness, the same question gives (almost) the same answer
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------- Call the model ----------
# invoke() sends one prompt and waits for the full reply
res = llm.invoke("Python kya hai? Ek line mein batao")

print(type(res))          # <class 'langchain_core.messages.ai.AIMessage'>, the reply is a message object, not a string
print(res.content)        # The actual answer text
print(res.usage_metadata) # Token counts for this call, e.g. {'input_tokens': 18, 'output_tokens': 40, 'total_tokens': 58, ...}