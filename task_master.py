from json import tool
import json

from langgraph.graph import StateGraph
from langchain_core.messages import SystemMessage,ChatMessage,ToolMessage
from langchain_ollama import ChatOllama
from typing import TypedDict


step_ai = ChatOllama(model="qwen2.5:7b")
repeat_schema ={
    "type": "object",
    "properties" : {
    "answer": {"type": "string", "enum": ["update", "repeat", "stop"]},
    "reason": {"type": "string"}
    },
    "required": ["answer","reason"]
}
check_ai = ChatOllama(model="qwen2.5:7b",format=repeat_schema,temperature=0.2)

class AIstate(TypedDict):
    goal : str
    steps : list[str]

    
def add_steps(state : AIstate)-> AIstate:
    """plans how to reach goal by adding steps one at a time"""
    sysmess =(
        "you are a planner who increase the steps one at a time"
        f"this is the end goal : {state["goal"]}"
        f"this are the previous steps : {state['steps']}"
        "find the next step and return it in one line sentence"
        "if the goal is reached just write the word END in it, no other things"
    )
    result = step_ai.invoke(sysmess).content
    state["steps"] = state["steps"] + [result]
    print(""+result)
    return state

def check_step(state:AIstate)->AIstate:
    """checks whether the steps are correct and updates the state accordingly"""
    sysmess = (
        f"Goal: {state['goal']}\n"
        f"Steps taken: {state['steps']}\n\n"
        "Evaluate strictly:\n"
        "- return 'stop' if the goal is FULLY and COMPLETELY achieved by the steps above\n"
        "- return 'repeat' if the last step is wrong or redundant\n"
        "- return 'update' ONLY if the goal is NOT yet achieved and last step is valid\n"
        f"Think: has the goal been 100% completed? If yes, stop."
    )    
    result =  json.loads(check_ai.invoke([SystemMessage(content=sysmess)]).content) 
    print("the result is : "+result["answer"] + " because " + result["reason"])
    if result["answer"] == "update":
        return state
    elif result["answer"] == "repeat":  
        state["steps"] = state["steps"][:-1]
        return state
    elif result["answer"] == "stop":
        state["steps"] = state["steps"] + ["END"]
        return state
