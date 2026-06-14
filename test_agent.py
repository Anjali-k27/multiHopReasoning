from agent_engine import master_agent

QUERY = (
    "Should Microsoft use its cash to acquire Alpha AI or NeuralNet Labs? "
    "Use financial data for Microsoft, find out who Alpha AI competes with, "
    "and check the documents for any weaknesses regarding the competitor to make your decision."
)

REQUIRED_TOOLS = {
    "execute_sql_financials",
    "search_unstructured_documents",
    "traverse_knowledge_graph",
}

REQUIRED_FACTS = [
    "$50 Billion",
    "NeuralNet Labs",
    "cloud compute overhead",
]

print("=" * 70)
print("RUNNING MULTI-HOP REACT AGENT TEST")
print("=" * 70)
print(f"\nQuery:\n  {QUERY}\n")
print("-" * 70)

response = master_agent(command=QUERY)
brief = response.comprehensive_executive_brief
trajectory = response.trajectory

tools_called = {
    v for k, v in trajectory.items()
    if k.startswith("tool_name_") and v != "finish"
}

print("\n[AGENT TRAJECTORY]")
step = 0

while True:
    thought_key = f"thought_{step}"
    tool_key    = f"tool_name_{step}"
    args_key    = f"tool_args_{step}"
    obs_key     = f"observation_{step}"
    if thought_key not in trajectory:
        break
    print(f"\n  Step {step + 1}")
    print(f"  Thought   : {trajectory[thought_key][:120]}...")
    print(f"  Tool      : {trajectory[tool_key]}")
    print(f"  Args      : {trajectory[args_key]}")
    if obs_key in trajectory:
        print(f"  Observation: {trajectory[obs_key]}")
    step += 1

print("\n[EXECUTIVE BRIEF]")
print(brief)

print("\n" + "=" * 70)
print("PASS / FAIL CRITERIA")
print("=" * 70)

all_tools_used = REQUIRED_TOOLS.issubset(tools_called)
print(f"\n[CRITERION 1] All three tools invoked: {'PASS ✅' if all_tools_used else 'FAIL ❌'}")
for name in sorted(REQUIRED_TOOLS):
    used = name in tools_called
    print(f"    {'✅' if used else '❌'}  {name}")

facts_present = all(fact.lower() in brief.lower() for fact in REQUIRED_FACTS)
print(f"\n[CRITERION 2] Required facts synthesised in brief: {'PASS ✅' if facts_present else 'FAIL ❌'}")
for fact in REQUIRED_FACTS:
    found = fact.lower() in brief.lower()
    print(f"    {'✅' if found else '❌'}  '{fact}'")

overall = all_tools_used and facts_present
print(f"\n{'=' * 70}")
print(f"OVERALL RESULT: {'✅  ALL TESTS PASSED' if overall else '❌  TESTS FAILED — refine docstrings/signature and re-run'}")
print("=" * 70)
