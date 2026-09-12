"""Brand visibility analysis on frozen evidence tables.

Lives outside ``evidence_eval`` and ``modeling`` on purpose: evidence manifests hash
every file in those packages, so analysis code here can change without invalidating
a frozen evidence version.
"""
