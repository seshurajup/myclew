import React from 'react';
import {AbsoluteFill, Img, useCurrentFrame, useVideoConfig, interpolate, spring} from 'remotion';
import Prism from 'prismjs';
import 'prismjs/components/prism-python';
import 'prismjs/components/prism-rust';
import {astryx, prismToAstryx, accentFor} from './astryxTheme';

// charEnd: pace-lock — by this segment's end, exactly charEnd chars are typed, so the code being
// typed is ALWAYS the code the voice is talking about (fixes typing running ahead of narration).
export type Segment = {start: number; end: number; text: string; charEnd?: number; speechEnd?: number};

// One output/media event: shown when typing passes afterLine (code output), or at `at` seconds
// (paper-figure mode). image = URL/dataURI; marks = normalized highlight boxes on the image
// (paper-with-markers use case); caption = small label under the media.
export type OutputEvent = {
  afterLine?: number;
  at?: number;
  text?: string;
  image?: string;
  caption?: string;
  marks?: {x: number; y: number; w: number; h: number}[];
};

export type ShortProps = {
  code: string;
  language: string;
  title: string;
  segments: Segment[]; // transcript, timed in seconds
  cps: number; // typing speed, chars/sec
  tailSeconds: number; // hold finished code at the end
  maxSeconds?: number; // Shorts length target (hard YouTube cap 180)
  outputs?: OutputEvent[]; // interpreter outputs / paper figures shown in the media pane
  hook?: string; // title-card text; the FIRST FRAME is the Short's de-facto thumbnail on YouTube
  accent?: string; // playlist accent name (astryxTheme.accents) — chrome only, never layout
};

// Flatten a Prism token stream to [{text, color}] spans
type Span = {text: string; color: string};
const flatten = (toks: (string | Prism.Token)[], inherit: string): Span[] => {
  const out: Span[] = [];
  for (const t of toks) {
    if (typeof t === 'string') out.push({text: t, color: inherit});
    else {
      const color = prismToAstryx[t.type] ?? inherit;
      const content = t.content as string | (string | Prism.Token)[];
      if (typeof content === 'string') out.push({text: content, color});
      else out.push(...flatten(content, color));
    }
  }
  return out;
};

// Cut highlighted spans to the first n typed characters
const takeChars = (spans: Span[], n: number): Span[] => {
  const out: Span[] = [];
  let left = n;
  for (const s of spans) {
    if (left <= 0) break;
    out.push(left >= s.text.length ? s : {...s, text: s.text.slice(0, left)});
    left -= s.text.length;
  }
  return out;
};

// Intro title card (0→1.4s, fades out). YouTube Shorts ignore custom thumbnails in the feed and
// show the first frame — so the first frame IS the poster: big hook text, astryx gradient.
const TitleCard: React.FC<{hook: string; language: string; ac: ReturnType<typeof accentFor>}> = ({hook, language, ac}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const t = frame / fps;
  // NO fade-in: frame 0 must be fully rendered, because YouTube takes frame 0 as the Shorts
  // poster. A 0.15s fade made the poster of every video a blank editor — the hook was invisible
  // exactly where it mattered most. Fade OUT only.
  const opacity = interpolate(t, [0, 1.05, 1.4], [1, 1, 0], {extrapolateRight: 'clamp'});
  // settle the pop instantly on frame 0 too, so the poster shows the hook at its final position
  const pop = frame === 0 ? 1 : spring({frame, fps, config: {damping: 200, stiffness: 120}});
  if (t >= 1.45) return null;
  const badge = ac.badge;
  return (
    <AbsoluteFill style={{background: ac.body, opacity, zIndex: 10, justifyContent: 'center',
                          alignItems: 'center', padding: 70}}>
      <div style={{position: 'absolute', inset: 0,
                   background: `radial-gradient(80% 55% at 50% 30%, ${ac.glowTop}, transparent), ` +
                               `radial-gradient(70% 50% at 50% 85%, ${ac.glowBottom}, transparent)`}} />
      <div style={{background: badge.bg, color: badge.fg, fontFamily: astryx.font.body, fontWeight: 800,
                   fontSize: 34, padding: '10px 30px', borderRadius: 999, marginBottom: 44,
                   transform: `scale(${0.9 + 0.1 * pop})`}}>{language}</div>
      <div style={{color: astryx.text, fontFamily: astryx.font.body, fontWeight: 800, fontSize: 84,
                   lineHeight: 1.15, textAlign: 'center', letterSpacing: -1,
                   transform: `translateY(${18 * (1 - pop)}px)`}}>{hook}</div>
      <div style={{color: astryx.textMuted, fontFamily: astryx.font.code, fontSize: 30, marginTop: 46}}>
        watch it typed &amp; run ▼</div>
    </AbsoluteFill>
  );
};

const TitleBar: React.FC<{title: string; language: string}> = ({title, language}) => {
  const badge =
    astryx.badge[language as keyof typeof astryx.badge] ?? astryx.badge.default;
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 18,
        padding: '26px 36px',
        background: astryx.surface,
        borderBottom: `2px solid ${astryx.border}`,
        borderRadius: '24px 24px 0 0',
      }}
    >
      {['#ffaeaa', '#eec12f', '#a6d2a2'].map((c) => (
        <div key={c} style={{width: 26, height: 26, borderRadius: 13, background: c}} />
      ))}
      <div
        style={{
          marginLeft: 12,
          color: astryx.textMuted,
          fontFamily: astryx.font.code,
          fontSize: 30,
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          flex: 1,
        }}
      >
        {title}
      </div>
      <div
        style={{
          background: badge.bg,
          color: badge.fg,
          fontFamily: astryx.font.body,
          fontWeight: 700,
          fontSize: 28,
          padding: '8px 22px',
          borderRadius: 999,
        }}
      >
        {language}
      </div>
    </div>
  );
};

const OutputPane: React.FC<{ev: OutputEvent; sinceFrame: number}> = ({ev, sinceFrame}) => {
  const {fps} = useVideoConfig();
  const frame = useCurrentFrame();
  const pop = spring({frame: frame - sinceFrame, fps, config: {damping: 200, stiffness: 260}});
  return (
    <div
      style={{
        background: astryx.codeBg,
        border: `2px solid ${astryx.border}`,
        borderRadius: 20,
        padding: '20px 26px',
        opacity: pop,
        transform: `translateY(${14 * (1 - pop)}px)`,
      }}
    >
      <div style={{color: astryx.textFaint, fontFamily: astryx.font.code, fontSize: 24, marginBottom: 10}}>
        {ev.image ? '▸ output' : '>>> output'}
      </div>
      {ev.text ? (
        <pre
          style={{
            margin: 0,
            fontFamily: astryx.font.code,
            fontSize: 30,
            lineHeight: 1.4,
            color: astryx.syntax.string,
            whiteSpace: 'pre-wrap',
            overflowWrap: 'break-word',
            paddingRight: 72,   // same right-action-rail inset as the code pane
          }}
        >
          {ev.text}
        </pre>
      ) : null}
      {ev.image ? (
        <div style={{position: 'relative', display: 'inline-block', maxWidth: '100%'}}>
          <Img src={ev.image} style={{maxWidth: '100%', maxHeight: 380, borderRadius: 12, display: 'block'}} />
          {(ev.marks ?? []).map((m, i) => (
            <div
              key={i}
              style={{
                position: 'absolute',
                left: `${m.x * 100}%`,
                top: `${m.y * 100}%`,
                width: `${m.w * 100}%`,
                height: `${m.h * 100}%`,
                border: `4px solid ${astryx.syntax.attribute}`,
                borderRadius: 8,
                boxShadow: '0 0 0 2000px rgba(10,10,10,0.25)',
                opacity: pop,
              }}
            />
          ))}
        </div>
      ) : null}
      {ev.caption ? (
        <div style={{color: astryx.textMuted, fontFamily: astryx.font.body, fontSize: 26, marginTop: 10}}>
          {ev.caption}
        </div>
      ) : null}
    </div>
  );
};

const Captions: React.FC<{segments: Segment[]; t: number}> = ({segments, t}) => {
  const {fps} = useVideoConfig();
  const frame = useCurrentFrame();
  let seg = segments.find((s) => t >= s.start && t < s.end);
  if (!seg) {
    // no segment is "active" — fill the gap (and the tail) with the most recent caption that has
    // started, so a caption is NEVER blank between segments. Only truly before the first one do we hide.
    for (let i = segments.length - 1; i >= 0; i--) {
      if (t >= segments[i].start) { seg = segments[i]; break; }
    }
    if (!seg) return null;
  }
  const pop = spring({frame: frame - seg.start * fps, fps, config: {damping: 200, stiffness: 300}});
  // karaoke: words light up as they are spoken — progress through the segment weighted by word
  // length (matches natural speech pacing closely enough without word-level timestamps)
  const words = seg.text.split(' ');
  const weights = words.map((w) => w.length + 1);
  const total = weights.reduce((a, b) => a + b, 0);
  // Highlight across the ACTUAL spoken window [start, speechEnd], NOT the padded seg.end —
  // seg.end includes silence added to stretch the video to its target length, and spreading the
  // karaoke over that silence made words light up after the voice already finished. speechEnd is
  // the real end of narration audio for this segment; fall back to seg.end for untimed transcripts.
  const voiceEnd = seg.speechEnd ?? seg.end;
  const rawProgress = (t - seg.start) / Math.max(voiceEnd - seg.start, 0.1);
  const progress = Math.pow(Math.max(0, rawProgress), 0.7); // slight front-load; all words lit by voiceEnd
  let acc = 0;
  const spoken = words.map((_, i) => {
    acc += weights[i];
    return progress >= acc / total - 0.03; // slightly more aggressive threshold
  });
  // FIT the caption inside its zone (top 130 → above the code at top 400): shrink the font by text
  // length so a long caption wraps to at most ~3 lines and NEVER grows down into the code block.
  const cl = seg.text.length;
  const capFont = cl > 150 ? 34 : cl > 108 ? 40 : cl > 70 ? 44 : 48;
  return (
    // Sit the caption band ABOVE YouTube's Shorts chrome: the bottom ~360px (progress bar,
    // channel row, title, description) and the right ~150px (like/comment/share buttons) are
    // covered by YouTube's UI on mobile — anything there is unreadable. So the band lives over
    // the lower code section, clear of both, with a solid backing so the code behind it never
    // bleeds through.
    <div
      style={{
        position: 'absolute',
        top: 120, left: 20, right: 20,
        height: 180,                            // FIXED caption zone (ends y300, clear of the titlebar)
        display: 'flex', flexDirection: 'column', justifyContent: 'center',
        textAlign: 'center', overflow: 'hidden',
        transform: `scale(${0.97 + 0.03 * pop})`,
        transformOrigin: 'top center',
      }}
    >
      <span
        style={{
          display: 'block',
          width: '100%',
          boxSizing: 'border-box',
          background: 'rgba(18,18,18,1)',
          border: `2px solid ${astryx.border}`,
          borderRadius: 22,
          padding: '26px 34px',
          fontFamily: astryx.font.body,
          fontSize: capFont,
          fontWeight: 800,
          lineHeight: 1.28,
          boxShadow: '0 10px 44px rgba(0,0,0,0.7)',
        }}
      >
        {words.map((w, i) => (
          <span key={i}
            style={{color: spoken[i] ? astryx.syntax.function : astryx.text,
                    transition: 'color 80ms'}}>
            {w}{i < words.length - 1 ? ' ' : ''}
          </span>
        ))}
      </span>
    </div>
  );
};

export const Short: React.FC<ShortProps> = ({code, language, title, segments, cps, outputs, hook, accent}) => {
  const ac = accentFor(accent);
  const frame = useCurrentFrame();
  const {fps, height, width} = useVideoConfig();
  const t = frame / fps;

  const grammar = Prism.languages[language] ?? Prism.languages.python;
  const spans = React.useMemo(
    () => flatten(Prism.tokenize(code, grammar), astryx.syntax.variable),
    [code, grammar],
  );
  // indentation = ONE keystroke (Tab), not N space chars: snap the typed position across each
  // line's leading whitespace so indents appear instantly, like a real coder pressing Tab.
  const indentRanges = React.useMemo(() => {
    const r: [number, number][] = [];
    let idx = 0;
    for (const ln of code.split('\n')) {
      const ws = ln.match(/^[ \t]+/);
      if (ws) r.push([idx, idx + ws[0].length]);
      idx += ln.length + 1;
    }
    return r;
  }, [code]);
  const snapIndent = (n: number): number => {
    for (const [a, b] of indentRanges) {
      if (n > a && n < b) return b;
      if (a >= n) break;
    }
    return n;
  };

  // pace-locked typing: piecewise-linear through (segment end, charEnd) anchors; uniform cps fallback
  const paced = segments.length > 0 && segments[0].charEnd != null;
  let nTyped: number;
  if (paced) {
    nTyped = 0;
    let prevChar = 0;
    for (const sg of segments) {
      if (t >= sg.end) { nTyped = sg.charEnd!; prevChar = sg.charEnd!; }
      else if (t >= sg.start) {
        nTyped = Math.round(prevChar + (sg.charEnd! - prevChar) * (t - sg.start) / (sg.end - sg.start));
        break;
      } else break;
    }
    if (t >= segments[segments.length - 1].end) nTyped = code.length; // tail: everything visible
    nTyped = Math.min(code.length, nTyped);
  } else {
    nTyped = Math.min(code.length, Math.floor(t * cps));
  }
  nTyped = Math.min(code.length, snapIndent(nTyped));
  const typed = takeChars(spans, nTyped);
  const cursorOn = Math.floor(t * 2.5) % 2 === 0 || nTyped < code.length;

  // active output event: latest one whose trigger (line typed, or time) has passed
  const lineEnds = React.useMemo(() => {
    const ends: number[] = [];
    let n = 0;
    for (const ln of code.split('\n')) { n += ln.length + 1; ends.push(n); }
    return ends;
  }, [code]);
  const charToSec = (c: number): number => {
    if (!(segments.length > 0 && segments[0].charEnd != null)) return c / cps;
    let prevChar = 0, prevEnd = 0;
    for (const sg of segments) {
      if (c <= sg.charEnd!) {
        const span = sg.charEnd! - prevChar;
        return sg.start + (sg.end - sg.start) * (span > 0 ? (c - prevChar) / span : 1);
      }
      prevChar = sg.charEnd!; prevEnd = sg.end;
    }
    return prevEnd;
  };
  const trigFrame = (ev: OutputEvent): number =>
    ev.at != null ? Math.round(ev.at * fps)
      : Math.round(charToSec(lineEnds[Math.min((ev.afterLine ?? 1) - 1, lineEnds.length - 1)]) * fps);
  // highlight the code the voice is CURRENTLY explaining: the active segment's char range glows
  let hiStart = -1, hiEnd = -1;
  if (paced) {
    // groupStart only advances when charEnd actually changes, so consecutive segments
    // pinned to the same until_line (one code line explained across 2+ sentences) keep
    // the SAME highlighted range instead of collapsing to a zero-width gap.
    let groupStart = 0;
    let prevCharEnd: number | undefined;
    for (const sg of segments) {
      if (prevCharEnd !== undefined && sg.charEnd !== prevCharEnd) groupStart = prevCharEnd;
      if (t >= sg.start && t < sg.end) { hiStart = groupStart; hiEnd = sg.charEnd!; break; }
      prevCharEnd = sg.charEnd;
    }
  }

  const fired = (outputs ?? []).filter((ev) => frame >= trigFrame(ev));
  const active = fired.length
    ? fired.reduce((a, b) => (trigFrame(a) >= trigFrame(b) ? a : b)) // chronologically latest
    : null;

  // AUTO-FIT: size line-height/font so the code fills the pane instead of leaving dead space.
  // short snippets get a big, readable font; long ones shrink to fit. Clamped both ways.
  const nCodeLines = code.split('\n').length;
  // reserve the lower ~920px so all code/output ends above the caption band (0.65-0.80H). The
  // caption sits above YouTube's Shorts chrome; keeping content out of that band also keeps the
  // tail frame clear, which verify_sync requires. +470 more when a media/output pane is shown.
  // while an output overlay is on screen, reserve ~560px so the code box ends ABOVE it and the
  // auto-scroll keeps the current line visible; otherwise code fills down to the CTA-safe zone.
  const outShrink = active
    ? spring({frame: frame - trigFrame(active), fps, config: {damping: 200, stiffness: 260}})
    : 0;
  // FIXED ZONES (stable → nothing jumps): caption 120..350, code starts at 366, safe bottom = H-480.
  // when the video has outputs, reserve a fixed output zone at the bottom so the code height is the
  // SAME for every frame (code font/position never change as it types → no jump, full readability).
  const CODE_TOP = 398;              // caption ends y300, titlebar 316..398, code card starts here
  const SAFE_BOT = height - 480;
  const hasOutputs = (outputs?.length ?? 0) > 0;
  const outHasImage = (outputs ?? []).some((o) => o.image);   // images need a taller zone than text
  // SMART BLOCK: the output zone is sized to its actual content (a one-line result != a 440px block),
  // so the code block keeps every pixel it doesn't need — code-first, output compact.
  const outLines = (outputs ?? []).reduce(
    (a, o) => a + (o.image ? 0 : Math.max(1, (o.text ?? '').split('\n').length)), 0);
  const OUT_ZONE = !hasOutputs ? 0
    : outHasImage ? 500
    : Math.min(260, 84 + 42 * outLines);
  const codeH = SAFE_BOT - CODE_TOP - (hasOutputs ? OUT_ZONE + 16 : 0);
  const contentH = codeH - 76;                       // inside the card padding
  // RIGHT-RAIL INSET: YouTube's like/comment/share buttons occupy the right ~10% below y0.40.
  // The card may extend under them (its background is dark), but no TEXT or caret may — a long
  // code line used to push the typing caret to x≈985, straight under the buttons. Wrapping the
  // text earlier keeps every glyph readable without narrowing the card visually.
  // Content runs x=58..1022; capping it at 950 leaves the rail (x>972) clear with margin.
  const RAIL_PAD = 72;
  // wrap-aware VISUAL line count at a given font (chars-per-line scales with the card width / font).
  const cplAt = (f: number) => Math.max(10, Math.floor((width - 40 - 76 - RAIL_PAD) / (f * 0.6)));
  const visAt = (s: string, f: number) => {
    const c = cplAt(f);
    return s.split('\n').reduce((a, ln) => a + Math.max(1, Math.ceil(ln.length / c)), 0);
  };
  // FIT the largest font in [38, 72] where every wrapped line is visible → code is fully STATIC
  // (no scroll, no jump) and fills the block. Only code too dense even at 38 falls back to scroll.
  let codeFont = 34, lineH = Math.round(34 / 0.72), scroll = 0, fits = false;
  for (let f = 72; f >= 34; f--) {
    const lh = Math.round(f / 0.72);
    if (visAt(code, f) * lh <= contentH) { codeFont = f; lineH = lh; fits = true; break; }
  }
  if (!fits) {                                       // genuinely long code: floor font 34, scroll to caret
    codeFont = 34; lineH = Math.round(34 / 0.72);
    const cvis = visAt(code.slice(0, nTyped), 34);
    scroll = Math.max(0, (cvis + 1) * lineH - contentH * 0.85);
  }

  return (
    <AbsoluteFill style={{background: ac.body, fontFamily: astryx.font.body}}>
      <AbsoluteFill style={{background:
        `radial-gradient(90% 45% at 50% 0%, ${ac.glowTop}, transparent), ` +
        `radial-gradient(90% 45% at 50% 100%, ${ac.glowBottom}, transparent)`}} />
      <Captions segments={segments} t={t} />
      {/* CODE block: fixed top + fixed height → stable, never jumps as code types in */}
      <div style={{position: 'absolute', top: CODE_TOP - 82, left: 20, right: 20}}>
        <TitleBar title={title} language={language} />
        <div
          style={{
            background: astryx.codeBg,
            borderRadius: '0 0 24px 24px',
            border: `2px solid ${astryx.border}`,
            borderTop: 'none',
            padding: '30px 38px',
            height: codeH,
            overflow: 'hidden',
          }}
        >
          <pre
            style={{
              margin: 0,
              fontFamily: astryx.font.code,
              fontSize: codeFont,
              fontWeight: 500,
              lineHeight: `${lineH}px`,
              whiteSpace: 'pre-wrap',
              overflowWrap: 'break-word',
              paddingRight: RAIL_PAD,   // keep glyphs + caret out of the right action-rail
              transform: `translateY(${-scroll}px)`,
            }}
          >
            {(() => {
              // per-LINE rendering: each code line is a block; the lines being narrated get a
              // clean full-width highlight (char-range span glow broke into jagged shapes on wrap)
              type Piece = {text: string; color: string};
              const lines: Piece[][] = [[]];
              const lineStart: number[] = [0];
              let off = 0;
              for (const sp of typed) {
                const chunks = sp.text.split('\n');
                for (let c = 0; c < chunks.length; c++) {
                  if (c > 0) { off += 1; lines.push([]); lineStart.push(off); }
                  if (chunks[c]) { lines[lines.length - 1].push({text: chunks[c], color: sp.color}); off += chunks[c].length; }
                }
              }
              return lines.map((ps, li) => {
                const a = lineStart[li];
                const b = li + 1 < lineStart.length ? lineStart[li + 1] - 1 : off;
                const hi = hiStart >= 0 && b > hiStart && a < hiEnd && a < b; // narrated, non-empty line
                return (
                  <div key={li} style={{minHeight: lineH,
                                        ...(hi ? {background: 'rgba(160,202,255,0.10)',
                                                  borderLeft: '6px solid rgba(160,202,255,0.55)',
                                                  marginLeft: -14, paddingLeft: 8,
                                                  borderRadius: 8} : {})}}>
                    {ps.map((pc, j) => (
                      <span key={j} style={{color: pc.color}}>{pc.text}</span>
                    ))}
                    {li === lines.length - 1 && cursorOn ? (
                      <span style={{display: 'inline-block', width: Math.round(codeFont * 0.45),
                                    height: Math.round(codeFont * 0.9),
                                    verticalAlign: 'text-bottom', background: astryx.text}} />
                    ) : null}
                  </div>
                );
              });
            })()}

          </pre>
        </div>
      </div>
      {active ? (
        <div style={{position: 'absolute', left: 20, right: 20, top: SAFE_BOT - OUT_ZONE,
                     height: OUT_ZONE, overflow: 'hidden',
                     opacity: outShrink, transform: `translateY(${14 * (1 - outShrink)}px)`}}>
          <OutputPane ev={active} sinceFrame={trigFrame(active)} />
        </div>
      ) : null}
      {hook ? <TitleCard hook={hook} language={language} ac={ac} /> : null}
    </AbsoluteFill>
  );
};
