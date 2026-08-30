# M6 real-world validation sequences

This directory holds M6's real-world validation clips -- used to check
whether the existing M1-M5 pipeline can consume real traffic footage and
to expose its actual limitations. Video files are gitignored; re-download
with the commands below.

**IMPORTANT -- experimental honesty note:** the sequences in this
directory (and `datasets/recorded/dev/car-detection.mp4`, reused
alongside them for M6) are NOT two views of the same physical corridor.
They are independent real-video sequences used for single-camera
validation, plus a mechanically labeled non-corridor stand-in for
exercising multi-camera interfaces. Never call them a real two-camera
corridor, paired corridor cameras, or ground truth for cross-camera
propagation.

## Sequence 2: `person-bicycle-car-detection.mp4`

- **Source:** https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4
- **Publisher/repo:** Intel IoT DevKit `sample-videos` (the same public GitHub repo already used for `datasets/recorded/dev/car-detection.mp4` in M1 -- no new source was searched for).
- **License:** CC BY 4.0 (Creative Commons Attribution 4.0 International), per the repo's `LICENSE` file: https://github.com/intel-iot-devkit/sample-videos/blob/master/LICENSE
- **Size:** 6,031,199 bytes (~5.75 MiB), 647 frames @ 12.0 fps (~53.9s), 768x432
- **SHA-256:** `452b11b7e0efbd019f1d9570d0c790e90416ad4ad29eec6003872d08443140ef`
- **Purpose:** M6 real-world validation only -- mixed vehicle/road-user traffic (pedestrians, bicycles, cars), the closest available "mixed traffic" content from this already-known source. Not part of any SUMO or paired-corridor evaluation track.

To re-fetch it locally:

```bash
curl -sL -o datasets/recorded/validation/person-bicycle-car-detection.mp4 \
  https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4
```

## Sequence 1 (reused from M1)

`datasets/recorded/dev/car-detection.mp4` -- see `datasets/recorded/dev/SOURCE.md` for its own provenance. Same publisher/repo, same CC BY 4.0 license.
