"""Live brand visibility advisor: a graph that measures, diagnoses, then advises.

Separate from ``visibility``, which reads the frozen evidence tables offline. This
package makes live calls (Serper for retrieval, Gemini for the assistant answer) and
turns what it observes into advice whose effect sizes come from the controlled test
in ``reports/intervention/`` -- never from a model's guess.

The dependency on ``langgraph`` lives only in ``graph.py`` and is installed at run
time (``uv run --with langgraph``), because ``uv.lock`` is hashed into every evidence
manifest and must not change.
"""
