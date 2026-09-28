"""CMMT benchmark-side helpers.

These modules aggregate experiment outputs and build dataset-specific manifests.
They intentionally do not own SAM 2 state export, translator implementation, or
RunPod tensor payloads; those remain in the translator runtime or local Task 07
orchestration workspace.
"""
