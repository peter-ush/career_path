from langchain_core.messages import HumanMessage, AIMessage
from career_agent import build_executor

def main():
    executor = build_executor(verbose=True)
    chat_history = []

    print("=== Career Chatbot (Agent Routing) ===")
    print("종료하려면 'exit' 입력\n")

    while True:
        user = input("You: ").strip()
        if user.lower() == "exit":
            break

        result = executor.invoke({
            "input": user,
            "chat_history": chat_history,
        })

        reply = result["output"]
        print(f"\nBot: {reply}\n")

        chat_history.append(HumanMessage(content=user))
        chat_history.append(AIMessage(content=reply))

if __name__ == "__main__":
    main()
