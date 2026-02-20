class ShortTermMemory:
    def __init__(self):
        self.chat_history = []

    def add(self, role, content):
        self.chat_history.append({"role": role, "content": content})

    def get_context(self):
        formatted = ""
        for msg in self.chat_history:
            formatted += f"{msg['role']}: {msg['content']}\n"
        return formatted
