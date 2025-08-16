import os 
from typing import Dict, TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from IPython.display import display, Image
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def load_conversation_history(filename: str):
    messages = []
    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("User:"):
                    messages.append(HumanMessage(content=line.replace("User:", "").strip()))
                elif line.startswith("AI:"):
                    messages.append(AIMessage(content=line.replace("AI:", "").strip()))
    except FileNotFoundError:
        # no file → return empty
        print(f"File {filename} not found. Starting with an empty conversation history.")
        return []
    except Exception as e:
        print(f"Error loading conversation history: {e}")
        return []
    
    return messages

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]  

llm = ChatOpenAI(model="gpt-4o-mini")

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state['messages'])
    print(response.content)
    state['messages'].append(AIMessage(content=response.content))

    print("Current state:", state["messages"])
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")  
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = load_conversation_history("conversation_history.txt")

user_input = input("Enter your message: ")
while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({
        "messages": conversation_history
    })
    conversation_history = result['messages']
    user_input = input("Enter your message (or type 'exit' to quit): ")

with open("conversation_history.txt", "w") as f:
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"User: {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"AI: {message.content}\n")
    f.write("\nEnd of conversation.\n")

print("Conversation history saved to conversation_history.txt")