from json import tool
import json
import pprint

from langgraph.graph import StateGraph,START,END
from langchain_core.messages import SystemMessage,ChatMessage,ToolMessage
from langchain_ollama import ChatOllama
from typing import TypedDict


step_ai = ChatOllama(model="qwen2.5:7b")
repeat_schema ={
    "type": "object",
    "properties" : {
    "answer": {"type": "string", "enum": ["update", "repeat", "stop"]},
    "reason": {"type": "string"},
    "next" : {"type": "string"}
    },
    "required": ["answer","reason","next"]
}
check_ai = ChatOllama(model="qwen2.5:7b",format=repeat_schema,temperature=0.2)

class AIstate(TypedDict):
    goal : str
    steps : list[str]

err = {"error":"", "reason":"" , "next":"","error":False}

    
def add_steps(state : AIstate)-> AIstate:
    """plans how to reach goal by adding steps one at a time"""
    global err
    sysmess = (
    f"You are a planner. Goal: {state['goal']}\n"
    f"Steps so far: {state['steps']}\n"
    "Return only the next single step in one line."
)
    if(err["error"]):
        sysmess += f"\nThe last step you provided was: '{err['error']}' and it was rejected because: {err['reason']}.\nPlease provide a better next step. Here's a suggestion for the next step: {err['next']}"
        err = {"error":"", "reason":"" , "next":"","error":False}
    print()
    result = step_ai.invoke([SystemMessage(content=sysmess)]).content
    state["steps"] = state["steps"] + [result]
    print("adder : added step : "+result)
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
    result =  json.loads(check_ai.invoke([SystemMessage(content=sysmess)]).content) 
    print()
    if result["answer"] == "update":
    
        print("checker : step is correct, keep going")
        return state
    elif result["answer"] == "repeat":  
        global err
        err = {"error":state["steps"][-1], "reason":result["reason"],"next":result["next"]}
        state["steps"] = state["steps"][:-1]
        print("checker : step is wrong, removing it and giving reason : "+result["reason"])
        return state
    elif result["answer"] == "stop":
        state["steps"] = state["steps"] + ["END"]
        print("checker : goal achieved, stopping")
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
lol = app.invoke({"goal": user_input, "steps": []})

pprint.pp(lol)
    