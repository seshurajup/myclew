# Video build inputs — Routing-Free Mixture-of-Experts

Prepared by `shorts-builder` (format=video) from this paper's lesson series.
Nothing here is rendered; this is the reviewable input set.

- `storyboard.json` — scenes in order (title/chapter/formula/figure/code/note)
- `props.json` — render props: canvas 1920×1080, safe zones, caption band, segment timings
- `narration.txt` — the spoken script, one block per caption segment
- `chapters.txt` — YouTube chapter markers
- `assets.txt` — every image the render needs (formula crops, figures, charts)

**Scenes** 64 · **segments** 64 · **assets** 51 (all present) · **script** ~903 words ≈ 6.0 min of speech

Build later:
```
kind=shorts-builder  spec={"format":"video","paper":"routing-free-moe","prefix":"rfm"}
```
