from typing import Annotated, Sequence, Dict, TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from IPython.display import display, Image
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.tools import tool 
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

document_content=""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
    """Update the document content."""
    global document_content
    document_content = content
    return f"Document updated with content: {content}"

@tool
def save(filename: str) -> str:
    """Save the document content to a file. Args: filename (str): The name of the text file to save the content to."""

    if filename.endswith('.txt'):
        filename = f"{filename}.txt"
    
    try: 
        global document_content
        with open(filename, 'w') as f:
            f.write(document_content)
        return f"Document saved to {filename}"
    
    except Exception as e:
        return f"An error occurred while saving the document: {str(e)}"

tools=[
    update,
    save,
]

model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content=f"""You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.

            - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
            - If the user wants to save and finish, you need to use the 'save' tool.
            - Make sure to always show the current document state after modifications.

            The current document content is: {document_content}
            """

    )

    if not state['messages']:
        user_input = "I'm ready to start drafting. what would you like me to do?"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\n What would you like me to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + state['messages'] + [user_message]
    print(f"All messages: {all_messages}")

    response = model.invoke(all_messages)
    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState) -> str:
    """Determine if the agent should continue based on the last message."""
    messages = state['messages']
    if not messages:
        return "continue"

    last_message = messages[-1]
    if isinstance(last_message, ToolMessage) and "saved" in last_message.content.lower() and "document" in last_message.content.lower():
        return "end"
    
    return "continue"

def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")

graph = StateGraph(AgentState)

graph.add_node("agent", model_call)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END
    },
)

app = graph.compile()

def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()
