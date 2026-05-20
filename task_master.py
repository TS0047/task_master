from json import tool
import json
import pprint

from langgraph.graph import StateGraph,START,END
from langchain_core.messages import SystemMessage,AIMessage,ToolMessage 
from langchain_ollama import ChatOllama
from typing import TypedDict


planner_ai = ChatOllama(model="qwen2.5:7b",temperature=0)
clarity_schema = {
    "type": "object",
    "properties": {
        "is_clear": {"type": "boolean"},
        "unclear_step_indices": {"type": "array", "items": {"type": "integer"}},
        "reason": {"type": "string"}
    },
    "required": ["is_clear", "unclear_step_indices", "reason"]
}
expand_schema = {
    "type": "object",
    "properties": {
        "expanded_steps": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"}
    },
    "required": ["expanded_steps", "reason"]
}
clarity_ai = ChatOllama(model="qwen2.5:7b", format=clarity_schema, temperature=0)
expand_ai = ChatOllama(model="qwen2.5:7b", format=expand_schema, temperature=0)

class AIstate(TypedDict):
    goal : str
    steps : list[str]
    iteration : int
    clarity_feedback : str
    unclear_indices : list[int]
    max_iterations : int


    
def generate_all_steps(state : AIstate) -> AIstate:
    """Generate all steps to reach the goal in a single call"""
    sysmess = (
        f"You are a planning assistant. Goal: {state['goal']}\n\n"
        "Generate a complete, ordered list of atomic steps needed to achieve this goal.\n"
        "Return ONLY a numbered list of steps, one per line.\n"
        "Be concise but specific.\n"
        "Do not number them in the response; just list them as separate items."
    )
    print("\n[generate_all_steps]")
    result = planner_ai.invoke([SystemMessage(content=sysmess)]).content
    
    # Parse steps from the response
    raw_steps = [line.strip() for line in result.strip().split('\n') if line.strip()]
    # Remove any numbering artifacts (e.g., "1.", "2.")
    steps = [step.lstrip('0123456789.-) ') for step in raw_steps]
    
    state["steps"] = steps
    state["iteration"] = 1
    state["clarity_feedback"] = ""
    state["unclear_indices"] = []
    state["max_iterations"] = 5
    
    print(f"Generated {len(steps)} steps:")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")
    
    return state


def evaluate_clarity(state : AIstate) -> AIstate:
    """Evaluate if the current steps are clear and unambiguous"""
    steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(state["steps"]))
    
    sysmess = (
        f"Goal: {state['goal']}\n\n"
        f"Current steps:\n{steps_text}\n\n"
        "Evaluate these steps for clarity and completeness:\n"
        "- 'is_clear': true if all steps are specific, unambiguous, and actionable\n"
        "- 'unclear_step_indices': array of 0-based indices of any vague or unclear steps\n"
        "- 'reason': brief explanation of what's unclear\n\n"
        "Return JSON format."
    )
    
    print(f"\n[evaluate_clarity] iteration {state['iteration']}")
    result = json.loads(clarity_ai.invoke([SystemMessage(content=sysmess)]).content)
    
    state["clarity_feedback"] = result["reason"]
    state["unclear_indices"] = result.get("unclear_step_indices", [])
    
    if result["is_clear"]:
        print(f"✓ All steps are clear. Ready to execute.")
        state["steps"] = state["steps"] + ["COMPLETE"]
    else:
        print(f"✗ Steps {state['unclear_indices']} lack clarity:")
        print(f"  {result['reason']}")
    
    return state


def expand_steps(state : AIstate) -> AIstate:
    """Expand the first unclear step for better clarity"""
    # Use the stored unclear indices from evaluate
    if not state["unclear_indices"]:
        return state
    
    # Expand the first unclear step
    idx = state["unclear_indices"][0]
    if idx >= len(state["steps"]):
        return state
    
    unclear_step = state["steps"][idx]
    
    sysmess = (
        f"Goal: {state['goal']}\n\n"
        f"The following step is unclear: '{unclear_step}'\n"
        f"Feedback: {state['clarity_feedback']}\n\n"
        "Break this step into 2-3 more specific sub-steps.\n"
        "Return as 'expanded_steps' array in JSON.\n"
        "Do NOT include the original step in the output."
    )
    
    print(f"\n[expand_steps] expanding step {idx+1}: '{unclear_step}'")
    result = json.loads(expand_ai.invoke([SystemMessage(content=sysmess)]).content)
    
    # Replace the unclear step with expanded ones
    expanded = result["expanded_steps"]
    state["steps"] = state["steps"][:idx] + expanded + state["steps"][idx+1:]
    state["iteration"] += 1
    state["unclear_indices"] = []  # Reset for next evaluation
    
    print(f"  → Expanded into {len(expanded)} sub-steps:")
    for sub in expanded:
        print(f"     • {sub}")
    
    return state

    
def router(state: AIstate) -> str:
    """Route based on clarity of steps and iteration limit"""
    # Check if we hit the COMPLETE marker
    if state["steps"] and state["steps"][-1] == "COMPLETE":
        return "end"
    
    # Check iteration limit
    if state["iteration"] >= state["max_iterations"]:
        print(f"\n⚠ Max iterations ({state['max_iterations']}) reached. Stopping expansion.")
        return "end"
    
    # If there are unclear steps, expand them
    if state["unclear_indices"]:
        return "expand"
    
    # Otherwise, re-evaluate
    return "evaluate"

graph = StateGraph(AIstate)

graph.add_node("generate", generate_all_steps)
graph.add_node("evaluate", evaluate_clarity)
graph.add_node("expand", expand_steps)

graph.add_edge(START, "generate")
graph.add_edge("generate", "evaluate")
graph.add_conditional_edges("evaluate", router, {"evaluate": "evaluate", "expand": "expand", "end": END})
graph.add_edge("expand", "evaluate")

app = graph.compile()
user_input = input("Enter your goal: ")
lol = app.invoke({
    "goal": user_input, 
    "steps": [], 
    "iteration": 0, 
    "clarity_feedback": "",
    "unclear_indices": [],
    "max_iterations": 5
})

print("\n" + "="*60)
print("FINAL STEPS:")
print("="*60)
for i, step in enumerate(lol["steps"], 1):
    if step != "COMPLETE":
        print(f"{i}. {step}")
print("="*60)

    