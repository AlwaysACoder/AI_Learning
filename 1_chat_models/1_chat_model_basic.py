# Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/

from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

# Load environment variables from .env
load_dotenv()

# Create a Mistral model
model = OllamaLLM(model="mistral")

# Invoke the model with a message
result = model.invoke("What is 5 * 3?")
print(result)