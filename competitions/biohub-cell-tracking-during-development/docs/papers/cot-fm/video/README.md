# Video build inputs — COT-FM: Cluster-wise Optimal Transport Flow Matching

Prepared by `shorts-builder` (format=video) from this paper's lesson series.
Nothing here is rendered; this is the reviewable input set.

- `storyboard.json` — scenes in order (title/chapter/formula/figure/code/note)
- `props.json` — render props: canvas 1920×1080, safe zones, caption band, segment timings
- `narration.txt` — the spoken script, one block per caption segment
- `chapters.txt` — YouTube chapter markers
- `assets.txt` — every image the render needs (formula crops, figures, charts)

**Scenes** 64 · **segments** 64 · **assets** 49 (all present) · **script** ~1509 words ≈ 10.1 min of speech

Build later:
```
kind=shorts-builder  spec={"format":"video","paper":"cot-fm","prefix":"cfm"}
```
