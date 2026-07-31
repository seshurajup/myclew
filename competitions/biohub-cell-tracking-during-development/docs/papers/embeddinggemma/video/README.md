# Video build inputs — EmbeddingGemma: Powerful and Lightweight Text Representations

Prepared by `shorts-builder` (format=video) from this paper's lesson series.
Nothing here is rendered; this is the reviewable input set.

- `storyboard.json` — scenes in order (title/chapter/formula/figure/code/note)
- `props.json` — render props: canvas 1920×1080, safe zones, caption band, segment timings
- `narration.txt` — the spoken script, one block per caption segment
- `chapters.txt` — YouTube chapter markers
- `assets.txt` — every image the render needs (formula crops, figures, charts)

**Scenes** 26 · **segments** 26 · **assets** 6 (all present) · **script** ~840 words ≈ 5.6 min of speech

Build later:
```
kind=shorts-builder  spec={"format":"video","paper":"embeddinggemma","prefix":"egm"}
```
