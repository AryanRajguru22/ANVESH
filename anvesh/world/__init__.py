"""World representation (blueprint Part 3, modules 2, 7, and 11): camera
calibration (image <-> world/road coordinates), corridor topology (camera
ordering, inter-camera distance), and corridor-state assembly (combining
per-camera TrafficStates into one CorridorState, M3).

No cross-camera cause-hypothesis fusion, cause reasoning, or temporal
alignment lives here -- alignment is `anvesh/fusion/alignment.py`; cause
fusion is `anvesh/fusion/ds_fusion.py`, still unimplemented (M4).
"""
