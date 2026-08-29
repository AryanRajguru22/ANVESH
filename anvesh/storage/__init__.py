"""Storage package boundary.

`schemas.py` holds the frozen Part 5 data contracts. `db.py` adds minimal
SQLite persistence behind small save/get/list interfaces, one per
concern, all in the same file/database: `CorridorStateStore` (M3),
`RankingStore` (M4), `FeedbackRecordStore` (M5) -- not a repository/ORM
layer.
"""
