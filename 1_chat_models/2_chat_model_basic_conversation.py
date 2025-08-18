from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import OllamaLLM

# Load environment variables from .env
load_dotenv()

# Create a Mistral model
model = OllamaLLM(model="mistral")

# SystemMessage:
#   Message for priming AI behavior, usually passed in as the first of a sequence of input messages.
# HumanMessagse:
#   Message from a human to the AI model.
messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 15 times 3?"),
]

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from AI: {result}")


# AIMessage:
#   Message from an AI.
messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 15 times 3?"),
    AIMessage(content="15 times 3 is 45."),
    HumanMessage(content="What is 10 times 5?"),
]

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from AI: {result}")
