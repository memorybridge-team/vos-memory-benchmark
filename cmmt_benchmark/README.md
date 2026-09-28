# CMMT benchmark helpers

This package is the benchmark-side home for MOSEv2/LVOS v2 manifest builders
and aggregate-only reporting helpers.  It consumes frozen manifests and
runtime reports; it does not store datasets, paired-state tensors, checkpoints,
or model secrets.

`vos-memory-translator-nonlinear` owns the reusable SAM 2 state contract and
runtime injection.  Local `CMMT/scripts/task07/` owns paired-state collection
and state-only development checks.  Task 08/13 evaluators can import these
helpers after their public protocol is implemented.
