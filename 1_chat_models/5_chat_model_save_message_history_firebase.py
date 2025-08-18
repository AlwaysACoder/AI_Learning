from dotenv import load_dotenv
from langchain_community.chat_message_histories.file import FileChatMessageHistory
from langchain_ollama import OllamaLLM


load_dotenv()


# Initialize File Chat Message History
print("Initializing File Chat Message History...")
chat_history = FileChatMessageHistory(file_path="chat_history.json")
print("Chat History Initialized.")
print("Current Chat History:", chat_history.messages)

# Initialize Chat Model
model = OllamaLLM(model="mistral")

print("Start chatting with the AI. Type 'exit' to quit.")

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break

    chat_history.add_user_message(human_input)

    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response)

    print(f"AI: {ai_response}")
