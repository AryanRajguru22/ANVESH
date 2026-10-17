"""Streaming single-camera world model (M-A, ANVESH_PRODUCT_GAP_AUDIT.md SS19/SS23).

One architecture for recorded files and live streams:

    FrameSource --(bounded queue, logged drops)--> StreamingPerceptionEngine
        detect+track -> StreamingTrackStore (session-scoped identity)
        -> SlidingWindowScheduler -> WindowSnapshot -> (WindowReasoner)

`anvesh.perception.pipeline.run_m1_pipeline` (the recorded-file batch replay)
is built on the same engine, so a recorded file takes exactly the path a live
stream takes.
"""
