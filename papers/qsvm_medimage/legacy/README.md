# Legacy code

This directory preserves imported upstream or local exploratory code that is
not used by the supported CPU, catalogue, protocol-matrix or notebook paths.
Files here are retained for attribution and provenance, not as an active API.

`qve_process.py` archives the intact implementations of `data_prepare` and
`process_folds` previously kept as commented-out blocks in `qve/process.py`.
They originated in the imported `sebasmos/qml-medimage` code and were removed
from the active module after repository-wide reference checks confirmed that
they were unused. The supported preprocessing entry point remains
`qve.process.data_prepare_cv`.
