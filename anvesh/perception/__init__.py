"""Single-camera perception pipeline (blueprint Part 3, modules 1, 3, 4, 5, 7, 8).

Video ingestion -> YOLO detection -> ByteTrack tracking -> track lifecycle
management -> image- or world-space motion measurement -> per-window
traffic-state aggregation (module 5, M3) -> conversion into the frozen
`anvesh.storage.schemas` contracts, plus a debug visualization utility.

No cross-camera fusion, cause hypotheses, propagation, or feedback live
here -- those start at M4+.
"""
