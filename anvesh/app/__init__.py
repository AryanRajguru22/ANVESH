"""ANVESH application layer: the operational product around the perception
core -- cameras and processing sessions, incidents, evidence, plate readings,
registry lookups, officer-reviewed violation cases, alert routing, audit, and
the HTTP API + dashboard that expose them.

Nothing here re-implements perception: sessions drive the existing
`anvesh.streaming` engine and consume its windows and stop episodes.
Everything that is not connected to a real external system (OCR engine,
vehicle registry, e-challan portal, alert channels) sits behind an adapter
whose status is reported honestly as `not_connected`/`sandbox`; see
`anvesh.app.capabilities`.
"""
