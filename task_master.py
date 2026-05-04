from langgraph.graph import StateGraph
from langchain_core.messages import SystemMessage,ChatMessage,ToolMessage
from langchain_ollama import OllamaLLM
from typing import TypedDict

model = OllamaLLM(model="mistral:7b")

class AIstate(TypedDict):
    goal : str
    steps : list[str]
    max : int


def add_steps(state : AIstate)-> AIstate:
    """plans how to reach goal by adding steps one at a time"""
    sysmess =(
        "you are a planner who increase the steps one at a time"
        f"this is the end goal : {state["goal"]}"
        f"this are the previous steps : {state['steps']}"
        f"the no of steps left is {state['max']}, so plan carefully"
        "find the next step and return it in one line sentence"
        "if the goal is reached just write the word END in it, no other things"
    )

    result = model.invoke(sysmess)
    print("Added step :"+result)
    state["steps"].append(result)
    state["max"] -= 1
    return state
    
lol = AIstate({"goal":"make me a sandwich","max" : 10 ,"steps":["START"]})

while lol["max"]>0:
    lol = add_steps(lol)
print(
    f"""
    goal :{lol['goal']}
    steps : {lol['steps']}
    """
    )