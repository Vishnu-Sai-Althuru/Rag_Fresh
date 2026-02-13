from short_memory import ShortTermMemory
from long_memory import LongTermMemory
from agent import generate_response

short_memory = ShortTermMemory()
long_memory = LongTermMemory()

print("AI Assistant with Memory (type 'exit' to stop)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    response = generate_response(user_input, short_memory, long_memory)

    print("AI:", response)

    # Store in short-term memory
    short_memory.add("User", user_input)
    short_memory.add("Assistant", response)

    # Store in long-term memory
    long_memory.add_memory(f"User: {user_input}\nAssistant: {response}")
