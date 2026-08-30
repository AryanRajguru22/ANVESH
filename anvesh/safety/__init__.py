"""Safety-intelligence package boundary (M7, blueprint-external).

Part 5 has no safety-event concept; this package is the first capability
built on top of the frozen perception schema rather than inside the
blueprint's own candidate-cause reasoning pipeline (M1-M6).

`stopped_vehicle.py` is the first (and, for M7, only) detector:
persistent stopped-vehicle detection from a single camera's own
`VehicleTrack`. It is deliberately NOT accident, wrong-way, or speeding
detection -- see its module docstring for the exact scope boundary.
"""
