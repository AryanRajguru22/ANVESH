"""Storage package boundary.

`schemas.py` holds the frozen Part 5 data contracts. `db.py` (M3) adds
minimal SQLite persistence for `CorridorState` records, behind a small
`CorridorStateStore` interface -- not a repository/ORM layer.
"""
