import os
import dspy
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")

# Initializing LM 
lm = dspy.LM(
    model="gemini/gemini-2.5-flash",
    temperature=0.0,
    max_tokens=2048,
)
dspy.configure(lm=lm)

------------------------- TOOLS ----------------------

def execute_sql_financials(entity_name: str) -> str:
    """Use ONLY for exact mathematical figures, market caps, revenue, or cash."""
    if "Microsoft" in entity_name:
        return "SQL RESULT: Microsoft Liquid cash available: $50 Billion."
    return f"OBSERVATION ERROR: No financial records for '{entity_name}'."

def search_unstructured_documents(semantic_query: str) -> str:
    """Use ONLY for qualitative information, CEO quotes, or strategic weaknesses."""
    if "NeuralNet" in semantic_query:
        return "DOCUMENT CHUNK: 'NeuralNet Labs suffers from massive cloud compute overhead.'"
    return "DOCUMENT CHUNK: No relevant qualitative data found."

def traverse_knowledge_graph(entity_name: str) -> str:
    """Use ONLY to understand multi-hop relationships or 'who competes with who'."""
    if "Alpha AI" in entity_name:
        return "GRAPH TOPOLOGY: (Alpha AI) --[COMPETES_WITH]--> (NeuralNet Labs)."
    return "GRAPH TOPOLOGY: Node isolated."

class AlphaFundAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.react_engine = dspy.ReAct(
            signature="strategic_command -> comprehensive_executive_brief",
            tools=[execute_sql_financials, search_unstructured_documents, traverse_knowledge_graph],
            max_iters=5,
        )

    def forward(self, command: str):
        return self.react_engine(strategic_command=command)

master_agent = AlphaFundAgent()
