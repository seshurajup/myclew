# YouTube Shorts Factory

Automated pipeline: **inputs in → finished YouTube Short out** — typed-code animation (Remotion),
cloned natural voice with human sounds, generated music, real program outputs, pace-locked captions,
verified sync — reviewed on the :9090 hub, uploaded (private) per video or per playlist on command.

Channel: `UC_jShs-8fh0MOk8NWGbJ_iA` · Hub: `http://gpu:9090` · DB: Postgres `kaggle_shorts`

---

## 1. Directory layout

```
~/kaggle/2026/youtube/
├── README.md                    ← this file
├── build_youtube.sh             ← the one entry point (see §4)
├── common/                      ← shared assets used by ALL videos
│   ├── voice/af_heart_ref.wav   ← narrator reference (Kokoro af_heart, 17s) for voice cloning
│   ├── music/                   ← (future) curated ACE-Step bed library per mood
│   └── brand/                   ← (future) logo, end-card, channel fonts
├── remotion_shorts/             ← the Remotion (React) renderer + worker scripts
│   ├── src/Short.tsx            ← composition: typing, captions, outputs, highlights, title card
│   ├── cbx_tts.py               ← Chatterbox Turbo batch-TTS worker (runs in voicebox/.venv-tts)
│   └── ace_music.py             ← ACE-Step music worker (runs in ACE-Step/.venv)
├── gallery/                     ← rendered mp4 + poster png + sidecar json (served by :9090 hub)
├── 01-learn-python/             ← playlist (build order = numeric prefix)
│   ├── 001-fibonacci/           ← one video = one folder, ALL its inputs inside
│   │   ├── spec.json            ← the agent spec (see §3)
│   │   ├── code.py              ← the tutorial code that gets typed
│   │   ├── transcript.json      ← narration lines w/ until_line pins + [chuckle]/hmm tags
│   │   ├── outputs.json         ← interpreter outputs / images / marks timeline
│   │   └── artifacts.py         ← optional: generates images in artifacts/ by RUNNING real code
│   ├── 002-list-comprehensions/
│   └── ...
├── 02-learn-pytorch/
└── 03-learn-transformer/
```

**Naming rules:** playlists `NN-learn-<topic>`, videos `NNN-<slug>`. Build order = python first,
then pytorch, then transformer (numeric sort does it).

## 2. The single agent

Everything is ONE agent: `fleet_agents/shorts_builder.py` (fleet source of truth =
`biohub-cell-tracking-during-development/fleet_agents/`; myclew is a read-only mirror).

Per run it does automatically: TTS (Chatterbox clone, Kokoro fallback) → caption retiming from real
speech lengths → ACE-Step music bed (cached) ducked under voice → pace-locked Remotion render
(1080×1920, CRF 18) → `verify_sync` measurement → poster export → gallery + sidecar + Postgres row
(incl. build_seconds) → hub shows it. Upload only on explicit "ready to push".

Hub: `python -m fleet_agents.shorts_builder --hub` (ThreadingHTTPServer :9090, path-routed
`/<playlist>/<video>`, no-store, 3 panels: playlists → videos → player).

## 3. Video spec (`spec.json`)

```jsonc
{
  "language": "python",             // prism/pygments lexer
  "title": "fibonacci.py",          // editor title-bar text
  "hook": "Write your first Python program in 60s",  // title card = poster = TITLE = line 1 of desc
  "playlist": "Learn Python",       // hub + YouTube playlist
  "target_seconds": 55,             // 60-120s REQUIRED (cap 180) — enough code + narration to teach one concept fully, not a 40s stub
  "tts_engine": "chatterbox",       // "chatterbox" (human sounds) | "kokoro" (clean/fast)
  "music_prompt": "lofi hip hop, chill study beat, instrumental",   // optional ACE-Step style
  "code_path": "code.py",
  "transcript_path": "transcript.json",   // [{text, until_line}] — tags like [chuckle] allowed
  "outputs_path": "outputs.json"          // [{after_line|at, text|image, caption, marks}]
}
```

Paths are resolved relative to the video folder by `build_youtube.sh`.

**Transcript**: `until_line` pins each narration line to the code it explains (pace-lock: typing
NEVER runs ahead of the voice). Chatterbox Turbo supports the FULL set of 9 paralinguistic tags —
`[laugh]` `[chuckle]` `[gasp]` `[cough]` `[sigh]` `[groan]` `[sniff]` `[shush]` `[clear throat]` —
use them mid-sentence (not stacked at fixed positions) and vary which ones appear per video so a
playlist binge doesn't sound templated; they render as real vocal sounds, not read-aloud text, and
are stripped from on-screen captions automatically. **Never put a tag in the FIRST segment** — the
opening second decides the swipe, and `[gasp] Variables are…` spends it on a noise instead of the
promise (gated: `check_cold_open`). Avoid plain filler words like "hmm," at the same
spot in every video — prefer the real tags for naturalness.

**Length = content, never silence.** Every video must run **60-120s**. Reach that with *real
narration* — write **8-10 segments** that teach the concept thoroughly — NOT by padding gaps.
The builder now caps the inter-segment pause at **0.8s** (`build_audio`): a short narration will
just produce a shorter video, it will NOT stretch dead air. Long silence with a frozen caption
reads as broken sync. Word-by-word karaoke highlighting is driven by each segment's `speechEnd`
(the real end of the voice clip), so all words are lit exactly when the voice stops, never after.
Avoid the word "human" in narration — say "people", "someone", or "you".

**Outputs**: because Python is an interpreter, show what the code DOES: `after_line` text outputs
(`>>> output` console), images (plots, formula renders, paper figures) with `caption` and `marks`
(normalized {x,y,w,h} highlight boxes — also the paper-explainer mechanism), or `at` seconds.
`artifacts.py` must generate images by RUNNING the actual tutorial code — outputs are never faked.

### 3.9 Retention rules (ENFORCED — the build refuses non-compliant videos)

Defined once in `tools/retention_rules.py`, enforced by `tools/gate.py` from `build_youtube.sh`
before any GPU work, and audited repo-wide by `tools/validate_videos.py`. These are not style
preferences; each one maps to a measured failure of the first 81 videos.

| # | Rule | Why | Check |
|---|---|---|---|
| 1 | **Hook is a promise, not a label.** No `Topic: description` shape, ≥4 words, ≤60 chars, must contain a verb / "you" / a question / a curiosity noun (bug, gotcha, trap…). | The hook is simultaneously the title card, the Shorts **thumbnail** (custom thumbs are ignored in the feed), the YouTube **title**, and line 1 of the description. `"Lists: ordered collections"` names a topic; it gives nobody a reason not to swipe. 79/81 of the originals were labels. | `check_hook` |
| 2 | **Clean cold open.** No `[gasp]`/`[groan]`/`[cough]` tag anywhere in segment 0. | Second 1 decides the swipe; a vocal noise spends it on nothing. 37/81 of the originals opened on a tag. | `check_cold_open` |
| 3 | **Tail = next-video pointer + one CTA.** Last segment must name the next video *and* ask for exactly one interaction (comment / follow / save). | Numbered series only binge if the video says where to go next; 1/81 of the originals did. Zero comments follows directly from 76/81 never asking. | `check_tail` |

**Series.** `SERIES` in `tools/apply_retention_rules.py` defines the independent binge chains. The
seven `NN-python-*` playlists are ONE numbered curriculum (001–081) so they thread as a single
chain; `01-learn-python` and `03-learn-transformer` are separate series with their own chains. The
last video of each series has nothing to point forward to, so it sends the viewer back to the start
of its own series instead — `SERIES_END_RE` accepts that as a valid tail.

Authoring a new video: put the spoken name of every video in `tools/topics.tsv` (used to phrase
"up next, …") and the hook in `tools/hooks.tsv`, then
`python tools/apply_retention_rules.py tools/hooks.tsv` writes the tails and cleans the cold opens
across the whole curriculum. It is idempotent — re-run it after inserting a video and every
"next up" pointer re-threads itself.

`tools/author_legacy.py` is the one-shot that brought the 21 pre-curriculum videos
(`01-learn-python`, `03-learn-transformer`) up to standard: they had 5–7 segments (22–40s) and
several `until_line`/`after_line` values pointing past the end of their `code.py`.

**Sizing narration:** the measured delivery rate across the built curriculum is **2.29 words/sec**
including inter-segment gaps. Divide your word count by that to predict duration before building —
the nominal 2.6 wps speech rate will under-estimate by roughly 12%.

**Not** a rule, deliberately: the `code_fills` layout check. It measures ink rows in a fixed
y0.24–0.66 band that includes the title bar and the reserved output zone, so it reads ~0.27 on
every video and reports `false` even when the code font is at its 72px maximum. It is informational
only (`verify_layout` excludes it from the hard gate) — do not "fix" the font because of it.

## 4. Build script

```bash
./build_youtube.sh all                     # every playlist, numeric order, one video at a time
./build_youtube.sh 01-learn-python         # one playlist
./build_youtube.sh 01-learn-python 001     # one video
FORCE=1 ./build_youtube.sh ...             # rebuild even if already in gallery
```

Per video: run `artifacts.py` (if present) → call the agent with the folder's spec → the agent does
the rest (gallery/PG/hub). Output name = `<playlist>-<video-dir>.mp4`. Idempotent: skips videos
already in the gallery unless FORCE=1.

## 4.5 Voice: clone once, or record for real

Narration was synthetic from the start — Chatterbox Turbo cloning `af_heart_ref.wav`, a Kokoro
*American female* voice. (`build_youtube.sh` never passed `voice_ref`, so every video used that
module default regardless of what the other build scripts set.) Two ways out, and they stack:

| | clone (`/me`) | record (`/record`) |
|---|---|---|
| your time | ~40s read, once | 86 min of speech (819 clips) |
| result | synthetic speech **in your voice** | **genuinely your voice** |
| AI-voice risk | reduced | eliminated |
| wins when both exist | — | `narration_dir` overrides `voice_ref` |

Neither needs timing work: captions and typing retime from real clip lengths (`build_audio`).

### The microphone needs HTTPS

Browsers expose `getUserMedia` only on a secure origin, so the recorder is dead on
`http://gpu:9090`. The hub therefore also listens on **`https://gpu:9443`** (self-signed cert from
`tools/make_cert.py`, SANs cover hostname + localhost + LAN + Tailscale IPs). Accept the warning
once — *Advanced → Proceed* — and the mic works. **9443, not 9090:** pointing `https://` at the
plain HTTP port gives `ERR_SSL_PROTOCOL_ERROR`.

Prefer no cert warning? Tunnel, and use a localhost origin (always treated as secure):

```bash
ssh -L 9443:localhost:9443 gpu     # then https://localhost:9443/me
ssh -L 9090:localhost:9090 gpu     # then http://localhost:9090/me   (plain, also fine)
# keep it alive on a flaky link:
ssh -N -L 9443:localhost:9443 -o ServerAliveInterval=30 gpu
```

### Cloning

`https://gpu:9443/me` shows a ~40s phonetically varied passage (`?mode=full` gives the 6-minute
curriculum read if you want a wider pool). Zero-shot cloning consumes only ~10–30s, so the short
read is the default. It records, plays back, reports duration/peak/RMS/SNR, refuses to install a
clipped, quiet, noisy or short take, picks the cleanest 30s window (vectorised — a 6-minute take
analyses in 0.08s), and on confirmation writes `common/voice/my_voice_ref.wav` plus
`my_voice_ref.txt` (the spoken transcript, for cloners that accept a `ref_text`) and points all 81
specs at it. Same thing from the CLI, and it accepts video too (PyAV extracts the audio; there is
no ffmpeg on this box):

```bash
python tools/set_voice.py check sample.mp4    # inspect, don't install
python tools/set_voice.py set   sample.wav    # install + point all 81 specs at it
python tools/set_voice.py show                # what is the narrator right now?
```

**Reference quality decides all 81 narrations.** Dead-silent room, one speaker, steady mic
distance, neutral pace, finish every word — a clip cut mid-word teaches a truncated phoneme, and
background hiss gets cloned along with the voice.

### Recording for real

`https://gpu:9443/record` lists every video with progress. Hold the button (or Space) per line,
play back, re-record. Completing a video's set flips its spec to `narration_dir` automatically;
if any clip is missing the whole video falls back to TTS, so a half-recorded video can never ship
half human and half synthetic. `python tools/narration.py status` reports coverage.

## 5. Engines & environments (all isolated — NEVER install into `llm` env)

| Piece | Where | Notes |
|---|---|---|
| Remotion 4 renderer | `remotion_shorts/` + Node 18 | `npx remotion render`, CRF 18, jpeg-q 95 |
| Chatterbox Turbo 350M (voice + non-verbals, MIT) | `external/voicebox/.venv-tts` | cu128 torch trio; patched: `resemble-perth` (not `perth`), float32 casts in `chatterbox/tts_turbo.py` + `s3tokenizer/utils.py` (torch 2.11 drops silent f64 promotion) |
| Kokoro-82M (fallback voice, Apache-2.0) | `llm` env | af_heart; also generated the clone reference |
| ACE-Step v1-3.5B music (Apache-2.0) | `external/ACE-Step/.venv` | 18.5s per 55s bed on 5090; cached in `config/_auto/music_cache` by (prompt,duration,seed); weights 7.8GB kept for new styles |
| voicebox (MIT, engine catalog + chunking reference) | `external/voicebox` | `backend/utils/chunked_tts.py` = long-form chunk/crossfade logic to lift when videos exceed ~1 sentence per segment |

## 6. YouTube upload

- OAuth: client secret `~/.google/oauth.json`, token `config/_auto/youtube_token.json` (auto-refresh)
- PRIVATE-only enforced in code; sync-gated (never uploads sync=FAIL); playlist auto-created
- Metadata auto-built (`build_metadata`): hook-first title ≤100ch + `#Shorts`, outcome-first
  description + hashtags, tags, category 27 (Education)
- Shorts classification is automatic: 9:16 ≤ 3min. First frame = poster (custom thumbs ignored
  in the Shorts feed) → the hook title card IS the thumbnail
- Upload per video or per playlist: user says "ready to push <video>" / "push playlist <name>"

## 7. Quality gates (all measured, never assumed)

- `verify_sync`: narration RMS inside vs outside caption windows (>1.5×) + caption-bubble pixels
  at every segment midpoint + clean tail frame → `sync=OK/FAIL` chip; FAIL blocks upload
- Pace-lock: `charEnd` anchors — typing position is a pure function of narration timeline
- Tab-indent: leading whitespace typed as ONE keystroke (real-coder feel)
- Karaoke captions: words light up as spoken (length-weighted timing)
- Frame inspection on every feature change (extract frames, look at them)

## 8. Future requirements / roadmap

1. **Full-bleed design** — remove side margins; gradient backdrop instead of flat black; richer
   editor chrome (line numbers, filename tab). *(in progress)*
2. **Author the full curriculum** — per playlist ~10 specs:
   - `01-learn-python`: fibonacci ✓, list comprehensions, decorators, generators, context
     managers, dataclasses, error handling, itertools, f-strings deep dive, walrus
   - `02-learn-pytorch`: tensors & autograd, one training step, DataLoader, nn.Module, GPU/bf16,
     optimizer anatomy, broadcasting, einsum
   - `03-learn-transformer`: attention ✓, multi-head, positional encoding, LayerNorm & residuals,
     the block, KV-cache, tokenization, why √d scaling
3. **Bed library** — pre-generate ~6 ACE-Step moods into `common/music/` (lofi, ambient, synthwave,
   jazz-hop, minimal piano, upbeat) so builds never wait on generation
4. **Voicebox chunked_tts adoption** — sentence-boundary chunking + 50ms crossfade + per-chunk
   deterministic seeds for segments longer than ~2 sentences (engine-agnostic; lift
   `backend/utils/chunked_tts.py`)
5. **Qwen3-TTS 1.7B option** — `instruct` prosody control ("warm, curious, slightly playful");
   A/B against Chatterbox for narration quality
6. **Longer formats** — 3-10min regular videos (16:9 composition variant) for deep dives; Shorts
   act as trailers linking to them
7. **Paper-explainer playlist** — outputs system already supports figures + marker boxes; add a
   PDF→figure-crop helper (pdffigures-style) and a `04-papers/` playlist
8. **Batch upload & scheduling** — "push playlist" command; YouTube `publishAt` scheduling so a
   whole playlist drip-releases daily
9. **Hub upgrades** — per-video retention notes field, A/B compare view (two renders side by side),
   playlist reorder, "rebuild" button writing a request file the build script picks up
10. **Analytics loop** — after videos go public: pull YouTube Analytics API (views, retention,
    swipe-away) into `kaggle_shorts` PG; correlate with video features (duration, cps, outputs
    count, music style) → data-driven content decisions
11. **Localization** — Chatterbox multilingual / Qwen3-TTS 10-langs: same video, re-voiced
    (hi/te/es) with translated captions
12. **End-card + subscribe animation** — last 1.5s branded outro from `common/brand/`
13. **Auto-QA voice** — Whisper-transcribe the final mix and diff against the transcript
    (catches TTS mispronunciations/skips automatically)
14. **Watch the durations** — Chatterbox generation is slower than Kokoro (~2-4s/segment after
    17s model load); consider persistent worker process if build times matter
```
