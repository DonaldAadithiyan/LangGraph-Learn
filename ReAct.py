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

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]  

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b

tools=[
    add,
]

model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are a helpful assistant. Use the tools provided to answer the user's questions."
    )
    response = model.invoke([system_prompt]+state['messages'])
    print(response.content)

    return {
        "messages": [response]}

def should_continue(state: AgentState) -> bool:
    messages = state['messages']
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)
graph.add_node("model_call", model_call)
graph.add_node("should_continue", should_continue)

tool_node= ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("model_call")

graph.add_conditional_edges(
    "model_call",
    should_continue,
    {
        "continue": "tools",
        "end": END
    },
)

graph.add_edge("tools", "model_call")

agent = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

input = {"messages": [("user", "What is 10 + 5?, what is 20 + 5?")]}
print_stream(agent.stream(input, stream_mode="values"))


