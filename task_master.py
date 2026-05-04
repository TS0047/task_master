from json import tool
import json
import pprint

from langgraph.graph import StateGraph,START,END
from langchain_core.messages import SystemMessage,AIMessage,ToolMessage 
from langchain_ollama import ChatOllama
from typing import TypedDict


step_ai = ChatOllama(model="qwen2.5:7b",temperature=0)
repeat_schema ={
    "type": "object",
    "properties" : {
    "answer": {"type": "string", "enum": ["update", "repeat", "stop"]},
    "reason": {"type": "string"},
    "next" : {"type": "string"}
    },
    "required": ["answer","reason","next"]
}
check_ai = ChatOllama(model="qwen2.5:7b",format=repeat_schema,temperature=0)

class AIstate(TypedDict):
    goal : str
    steps : list[str]
    chat : list[AIMessage]


    
def add_steps(state : AIstate)-> AIstate:
    """plans how to reach goal by adding steps one at a time"""
    sysmess = (
    f"You are a planner. Goal: {state['goal']}\n"
    f"Steps confirmed so far: {state['steps']}\n"
    "The chat history contains feedback from the checker if any step was rejected.\n"
    "If the checker suggested a 'next' step, use that as your next suggestion.\n"
    "Return only the next single step in one line. Never return an empty response."
)
    print()
    result = step_ai.invoke([SystemMessage(content = sysmess)]+state["chat"]).content
    state["chat"] = state["chat"] + [AIMessage(content=result,name="step_ai")]
    print(f"step_ai : i suggest [{result}] as the next step")
    return state

def check_step(state:AIstate)->AIstate:
    """checks whether the steps are correct and updates the state accordingly"""
    sysmess = (
        f"Goal: {state['goal']}\n"
        f"All steps so far: {state['steps']}\n\n"
        "Evaluate the LAST step only in context of the full plan:\n"
        "- 'update' if the last step is a valid, non-redundant step toward the goal\n"
        "- 'repeat' if the last step is wrong, redundant, or contradicts previous steps\n"
        "- 'stop' if ALL steps together FULLY complete the goal\n\n"
        "IMPORTANT: A single step does NOT need to complete the goal alone. "
        "It just needs to be a valid next step in the plan."
        "Respond in JSON format with 'answer' as one of 'update', 'repeat', or 'stop', and provide a brief 'reason' for your decision. If 'repeat', also suggest a 'next' step to replace the incorrect one."
    ) 
    result =  json.loads(check_ai.invoke([SystemMessage(content = sysmess)]+state["chat"]).content) 
    print()
    if result["answer"] == "update":
        print("checker : step is correct, keep going")
        state["steps"] = state["steps"] + [state["chat"][-1].content]


    elif result["answer"] == "repeat":
        state["chat"] = state["chat"] + [AIMessage( 
        content="Rejected: " + result["reason"] + ". Try instead: " + result["next"],
        name="check_ai"
    )]
        print("checker : step is wrong, removing it and giving reason : "+result["reason"]+"\n and suggesting next step : "+result["next"])  


    elif result["answer"] == "stop":
        state["steps"] = state["steps"] + ["END"]
        print("checker : goal achieved, stopping")

    if (len(state["chat"]) > 4):
        state["chat"] = state["chat"][-4:]
    return state
    
def router(state:AIstate)->str:
    """decides whether to add more steps or check the current steps"""
    if len(state["steps"]) == 0:
        return "add"
    if state["steps"][-1] == "END":
        return "end"
    return "add"

graph = StateGraph(AIstate)

graph.add_node("add", add_steps)
graph.add_node("check", check_step)
graph.add_node("router", lambda state:state)


graph.add_edge(START,"router")
graph.add_conditional_edges("router",router,{"add":"add","end":END})
graph.add_edge("add","check")
graph.add_edge("check","router")

app = graph.compile()
user_input = input("Enter your goal: ")
lol = app.invoke({"goal": user_input, "steps": [],"chat":[]})

pprint.pp(lol)
    