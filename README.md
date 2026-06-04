# 🧠 Task Master — Self-Refining Agentic Planner

> Give it a goal. It plans, critiques its own plan, and rewrites the vague parts until every step is actionable.

Task Master is a **local agentic planning system** built on **LangGraph** and **Ollama**. Instead of dumping a single rough plan, it runs an iterative *generate → critique → refine* loop: an LLM drafts steps, a second LLM judges their clarity, and unclear steps are automatically expanded into concrete sub-steps — until the plan is fully actionable or an iteration limit is hit.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=flat-square&logo=ollama&logoColor=white)
![Offline](https://img.shields.io/badge/100%25-Offline-success?style=flat-square)

---

## The Agent Graph

```
        ┌──────────┐
START → │ generate │  draft a complete ordered list of atomic steps
        └────┬─────┘
             ▼
        ┌──────────┐      is_clear? ──── yes ──→ END
        │ evaluate │  ◄──────────┐
        └────┬─────┘             │
             │ unclear           │ re-check
             ▼                   │
        ┌──────────┐             │
        │  expand  │ ────────────┘
        └──────────┘  break vague step into 2–3 sub-steps
```

A conditional router drives the loop:
- **All steps clear** → append a `COMPLETE` marker and finish
- **Unclear steps remain** → expand the first one, then re-evaluate
- **Max iterations reached** → stop gracefully

---

## Why It's Interesting

- **Three specialized LLM roles** — a *planner*, a *clarity critic*, and an *expander*, each a separate `ChatOllama` instance
- **Structured JSON output** — the critic and expander use enforced JSON schemas (`format=...`) so their decisions are machine-parseable, not free text
- **Self-correction** — the system reasons about the quality of its own output and iterates, rather than trusting the first draft
- **Deterministic** — `temperature=0` throughout for reproducible plans
- **Fully local** — runs on `qwen2.5:7b` via Ollama, no cloud calls

---

## Setup

```bash
# 1. Install Ollama and pull the model
ollama pull qwen2.5:7b

# 2. Install dependencies
pip install langgraph langchain-core langchain-ollama

# 3. Run it
python task_master.py
```

Then enter a goal when prompted, e.g. `Plan a weekend trip to the mountains` or `Set up a CI pipeline for a Python project`.

---

## Example

```
Enter your goal: bake a sourdough loaf

[generate_all_steps]
Generated 6 steps:
  1. Prepare the starter
  2. Mix the dough
  ...

[evaluate_clarity] iteration 1
✗ Steps [0] lack clarity:
  "Prepare the starter" is vague — no timing or quantities given

[expand_steps] expanding step 1: 'Prepare the starter'
  → Expanded into 3 sub-steps:
     • Feed the starter with equal parts flour and water 8 hours before mixing
     • Verify it has doubled and passes the float test
     • Measure out the required amount for the recipe

✓ All steps are clear. Ready to execute.
```

---

## Configuration

| Knob | Where | Default |
|------|-------|---------|
| Model | top of `task_master.py` | `qwen2.5:7b` |
| Max refinement iterations | `max_iterations` in initial state | `5` |

---

## Roadmap

- [ ] Execute the finalized steps with tool-calling agents
- [ ] Persist plans to disk / resume sessions
- [ ] Web UI for interactive plan editing
