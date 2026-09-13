# pyright: reportMissingImports=false
"""The graph. Installed at run time, because ``uv.lock`` is hashed into the manifests.

The shape is the argument of the product. First find out who competes, from the
results themselves, so any sector works. Then measure and score with the signal
learned on the recorded sectors. Then branch, because the situations a brand can be
in have different answers and only some of them are fixable with content.

    plan → search → discover → interrogate → analyse ─┬─ absent   ─┐
                                                      ├─ low_rank ─┤
                                                      ├─ ceiling  ─┼→ report
                                                      ├─ leader   ─┤
                                                      └─ thin     ─┘
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from . import nodes
from .state import AdvisorInput, AdvisorState

BRANCHES = {
    "absent": nodes.advise_absent,
    "low_rank": nodes.advise_low_rank,
    "ceiling": nodes.advise_ceiling,
    "leader": nodes.advise_leader,
    "thin": nodes.advise_thin,
}


def build(runtime: nodes.Runtime, budget: int):
    """Compile the advisor graph around one runtime (clients, receipts, model, budget)."""
    builder = StateGraph(AdvisorState, input_schema=AdvisorInput)
    builder.add_node("plan", nodes.make_plan(runtime, budget))
    builder.add_node("search", nodes.make_search(runtime, budget))
    builder.add_node("discover", nodes.make_discover(runtime, budget))
    builder.add_node("interrogate", nodes.make_interrogate(runtime, budget))
    builder.add_node("analyse", nodes.make_analyse(runtime))
    for name, node in BRANCHES.items():
        builder.add_node(f"advise_{name}", node)
    builder.add_node("report", nodes.report)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "search")
    builder.add_edge("search", "discover")
    builder.add_edge("discover", "interrogate")
    builder.add_edge("interrogate", "analyse")
    builder.add_conditional_edges(
        "analyse", nodes.route, {name: f"advise_{name}" for name in BRANCHES}
    )
    for name in BRANCHES:
        builder.add_edge(f"advise_{name}", "report")
    builder.add_edge("report", END)
    return builder.compile()
