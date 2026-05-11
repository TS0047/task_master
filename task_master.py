import json
import pprint
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from typing import TypedDict, Optional

# ── models ────────────────────────────────────────────────────────────────────

step_ai  = ChatOllama(model="qwen2.5:7b", temperature=0)

plan_schema = {
    "type": "object",
    "properties": {
        "milestones": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Ordered high-level phases the plan must pass through"
        },
        "done_when": {
            "type": "string",
            "description": "Precise, observable condition that marks the goal as fully complete"
        },
        "max_steps": {
            "type": "integer",
            "description": "Rough upper bound on total steps expected"
        }
    },
    "required": ["milestones", "done_when", "max_steps"]
}

check_schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["update", "repeat", "stop"]},
        "reason": {"type": "string"},
        "next":   {"type": "string"},
    },
    "required": ["answer", "reason", "next"],
}

plan_ai  = ChatOllama(model="qwen2.5:7b", format=plan_schema,  temperature=0)
check_ai = ChatOllama(model="qwen2.5:7b", format=check_schema, temperature=0)


# ── state ─────────────────────────────────────────────────────────────────────

class AIstate(TypedDict):
    goal:            str
    milestones:      list[str]   # set once by plan_agent
    done_when:       str         # set once by plan_agent
    max_steps:       int         # set once by plan_agent
    steps:           list[str]
    last_suggestion: str
    last_rejection:  Optional[str]
    done:            bool


# ── nodes ─────────────────────────────────────────────────────────────────────

def plan_agent(state: AIstate) -> AIstate:
    """Runs once. Defines milestones, completion criteria, and step budget."""
    sysmess = (
        f"Goal: {state['goal']}\n\n"
        "You are a planning assistant. Analyse the goal and output:\n"
        "  - milestones: ordered list of high-level phases needed\n"
        "  - done_when: a precise, observable sentence describing when the goal is 100%% complete\n"
        "  - max_steps: realistic upper bound on the number of atomic steps required\n"
        "Be specific. 'done_when' must be unambiguous — a checklist-style condition."
    )
    result = json.loads(plan_ai.invoke([SystemMessage(content=sysmess)]).content)

    print("\n=== Plan ===")
    print(f"Milestones : {result['milestones']}")
    print(f"Done when  : {result['done_when']}")
    print(f"Max steps  : {result['max_steps']}\n")

    return {
        **state,
        "milestones": result["milestones"],
        "done_when":  result["done_when"],
        "max_steps":  result["max_steps"],
    }


def add_steps(state: AIstate) -> AIstate:
    steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(state["steps"])) or "None yet."

    prompt = (
        f"Goal: {state['goal']}\n"
        f"Plan milestones: {state['milestones']}\n"
        f"Goal is complete when: {state['done_when']}\n\n"
        f"Confirmed steps so far:\n{steps_text}\n\n"
    )
    if state["last_rejection"]:
        prompt += f"Your last suggestion was rejected. Feedback: {state['last_rejection']}\n\n"

    prompt += (
        "Output ONLY the single next atomic step not yet in the confirmed list.\n"
        "Rules:\n"
        "- Never output empty text\n"
        "- Never repeat or rephrase a confirmed step\n"
        "- One sentence, no numbering, no preamble\n"
        "- If all milestones are covered by confirmed steps, output: DONE"
    )

    suggestion = step_ai.invoke([SystemMessage(content=prompt)]).content.strip()

    if not suggestion:
        suggestion = f"Review progress toward: {state['goal']}"

    print(f"\nstep_ai  : [{suggestion}]")
    return {**state, "last_suggestion": suggestion, "last_rejection": None}


def check_step(state: AIstate) -> AIstate:
    steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(state["steps"])) or "None yet."

    # Hard cap: force stop if over budget
    if len(state["steps"]) >= state["max_steps"] or state["last_suggestion"] == "DONE":
        print("checker  : ✓ step budget reached / planner signalled DONE — stopping")
        return {**state, "done": True}

    sysmess = (
        f"Goal: {state['goal']}\n"
        f"Goal is complete when: {state['done_when']}\n\n"
        f"Confirmed steps so far:\n{steps_text}\n\n"
        f"Proposed next step: \"{state['last_suggestion']}\"\n\n"
        "Evaluate this proposed step:\n"
        "  'update' → valid, non-redundant, moves plan forward\n"
        "  'repeat' → redundant, already covered by a confirmed step, or contradicts the plan\n"
        "  'stop'   → confirmed steps + this step satisfy the done_when condition exactly\n\n"
        "Key rule for 'repeat': if any confirmed step already covers the same action "
        "(even with different wording), answer 'repeat'.\n"
        "Return JSON with 'answer', 'reason', and 'next' (replacement if repeating)."
    )

    result = json.loads(check_ai.invoke([SystemMessage(content=sysmess)]).content)
    answer = result["answer"]

    if answer == "update":
        print(f"checker  : ✓ accepted  — {result['reason']}")
        return {**state, "steps": state["steps"] + [state["last_suggestion"]]}

    elif answer == "repeat":
        feedback = f"Rejected (redundant): {result['reason']}. Try instead: {result['next']}"
        print(f"checker  : ✗ rejected  — {result['reason']}")
        print(f"           → suggestion: {result['next']}")
        return {**state, "last_rejection": feedback}

    elif answer == "stop":
        print(f"checker  : ✓ goal complete — {result['reason']}")
        return {**state, "steps": state["steps"] + [state["last_suggestion"]], "done": True}

    return state


# ── routing ───────────────────────────────────────────────────────────────────

def router(state: AIstate) -> str:
    return "end" if state.get("done") else "add"


# ── graph ─────────────────────────────────────────────────────────────────────

graph = StateGraph(AIstate)
graph.add_node("plan",  plan_agent)
graph.add_node("add",   add_steps)
graph.add_node("check", check_step)

graph.add_edge(START, "plan")
graph.add_edge("plan", "add")
graph.add_edge("add",  "check")
graph.add_conditional_edges("check", router, {"add": "add", "end": END})

app = graph.compile()

# ── run ───────────────────────────────────────────────────────────────────────

user_input = input("Enter your goal: ")
result = app.invoke({
    "goal":            user_input,
    "milestones":      [],
    "done_when":       "",
    "max_steps":       20,
    "steps":           [],
    "last_suggestion": "",
    "last_rejection":  None,
    "done":            False,
})

print("\n=== Final Plan ===")
for i, step in enumerate(result["steps"], 1):
    print(f"  {i}. {step}")
print(f"\nDone when: {result['done_when']}")