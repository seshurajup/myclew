"""Browser recording booth for the :9090 hub — read the lines, get a real-voice video.

The narration for all 81 videos is synthetic. Recording 810 clips by hand with a file manager is
the kind of chore that never gets done, so this puts the whole loop in the browser: the hub shows
each line, you hold a key and read it, and the clip lands in the right folder with the right name.

Served by fleet_agents/shorts_builder.py's hub Handler:
    GET  /record                       index: every video and its recording progress
    GET  /record/<playlist>/<video>    the booth
    POST /api/narration?...            one clip (16-bit PCM WAV body)

WAV is encoded in the browser rather than server-side, because ffmpeg is not installed on this
machine and MediaRecorder's webm/opus would otherwise need transcoding.
"""
import json
import re
import struct
from pathlib import Path

YT = Path(__file__).resolve().parent.parent
PLAYLISTS = ["01-python-basics", "02-python-functions", "03-python-loops-iteration",
             "04-python-oop", "05-python-advanced", "06-python-testing-tools", "07-python-libraries"]
TAG = re.compile(r"\[[a-z ]+\]")
SAFE = re.compile(r"^[0-9A-Za-z._-]+$")


def _vdir(playlist, video):
    if not (SAFE.match(playlist or "") and SAFE.match(video or "")):
        raise ValueError("bad path")
    d = (YT / playlist / video).resolve()
    if not str(d).startswith(str(YT.resolve())) or not (d / "transcript.json").exists():
        raise ValueError("no such video")
    return d


def progress(d: Path):
    segs = json.loads((d / "transcript.json").read_text())
    nd = d / "narration"
    have = sum(1 for i in range(len(segs)) if (nd / f"{i + 1:03d}.wav").exists())
    return have, len(segs)


def save_clip(playlist, video, index, wav_bytes):
    """Persist one clip; when the set completes, switch the spec over to the real voice."""
    d = _vdir(playlist, video)
    segs = json.loads((d / "transcript.json").read_text())
    i = int(index)
    if not 1 <= i <= len(segs):
        raise ValueError("index out of range")
    if wav_bytes[:4] != b"RIFF":
        raise ValueError("not a WAV upload")
    nd = d / "narration"
    nd.mkdir(exist_ok=True)
    (nd / f"{i:03d}.wav").write_bytes(wav_bytes)
    have, want = progress(d)
    complete = have == want
    if complete:
        p = d / "spec.json"
        s = json.loads(p.read_text())
        if s.get("narration_dir") != "narration":
            s["narration_dir"] = "narration"
            p.write_text(json.dumps(s, indent=2) + "\n")
    return {"ok": True, "have": have, "want": want, "complete": complete}


def index_html():
    rows = []
    for pl in PLAYLISTS:
        base = YT / pl
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            if not (d.is_dir() and (d / "transcript.json").exists()):
                continue
            have, want = progress(d)
            spec = json.loads((d / "spec.json").read_text())
            pct = int(100 * have / want) if want else 0
            state = "done" if have == want else ("part" if have else "none")
            rows.append(
                f'<a class="row {state}" href="/record/{pl}/{d.name}">'
                f'<span class="n">{d.name}</span>'
                f'<span class="h">{spec.get("hook", "")}</span>'
                f'<span class="p"><i style="width:{pct}%"></i></span>'
                f'<span class="c">{have}/{want}</span></a>')
    return _PAGE.replace("__TITLE__", "narration booth").replace("__BODY__", f"""
      <h1>narration booth</h1>
      <p class="sub">Record your own voice. Captions and typing retime to your real speech —
         read at whatever pace sounds natural.</p>
      <div class="list">{''.join(rows)}</div>""").replace("__EXTRA__", "")


def booth_html(playlist, video):
    d = _vdir(playlist, video)
    segs = json.loads((d / "transcript.json").read_text())
    spec = json.loads((d / "spec.json").read_text())
    have, want = progress(d)
    lines = [{"i": i + 1, "text": " ".join(TAG.sub(" ", s["text"]).split())}
             for i, s in enumerate(segs)]
    body = f"""
      <a class="back" href="/record">&larr; all videos</a>
      <h1>{video}</h1>
      <p class="sub">{spec.get('hook','')} &middot; <b id="done">{have}</b>/{want} recorded</p>
      <p class="hint">Hold <kbd>Space</kbd> (or the button) to record a line. Release to stop.
         Play it back, re-record if you fluffed it. Leave a beat of silence at each end —
         it gets trimmed.</p>
      <div id="lines"></div>"""
    extra = f"""<script>
const PL={json.dumps(playlist)}, VID={json.dumps(video)}, LINES={json.dumps(lines)};
const HAVE={json.dumps([i + 1 for i in range(want) if (d / 'narration' / f'{i+1:03d}.wav').exists()])};
</script><script>{_JS}</script>"""
    return _PAGE.replace("__TITLE__", f"record {video}").replace("__BODY__", body).replace("__EXTRA__", extra)


def encode_wav(samples, sr=24000):
    """Server-side WAV writer (used by tests); the browser has its own copy in JS."""
    n = len(samples)
    hdr = (b"RIFF" + struct.pack("<I", 36 + n * 2) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
           + b"data" + struct.pack("<I", n * 2))
    return hdr + b"".join(struct.pack("<h", max(-32768, min(32767, int(s * 32767)))) for s in samples)


_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>
 :root{--bg:#1b1b1b;--sf:#262626;--bd:#404040;--tx:#e5e5e5;--mu:#a3a3a3;--ac:#a0caff;--gd:#a6d2a2;--rd:#ffaeaa}
 *{box-sizing:border-box} body{margin:0;padding:28px;background:var(--bg);color:var(--tx);
   font:16px/1.5 Figtree,system-ui,sans-serif}
 h1{margin:0 0 6px;font-size:26px} .sub{color:var(--mu);margin:0 0 14px}
 .hint{background:var(--sf);border:1px solid var(--bd);border-radius:10px;padding:12px 14px;color:var(--mu);margin:0 0 18px}
 kbd{background:#333;border:1px solid #555;border-radius:5px;padding:1px 7px;color:var(--tx)}
 a{color:var(--ac);text-decoration:none} .back{display:inline-block;margin-bottom:10px}
 .list{display:flex;flex-direction:column;gap:6px}
 .row{display:grid;grid-template-columns:230px 1fr 120px 60px;gap:12px;align-items:center;
   background:var(--sf);border:1px solid var(--bd);border-radius:9px;padding:9px 12px;color:var(--tx)}
 .row .n{font-family:ui-monospace,monospace;color:var(--ac)} .row .h{color:var(--mu)}
 .row .p{background:#111;border-radius:5px;height:8px;overflow:hidden} .row .p i{display:block;height:100%;background:var(--gd)}
 .row .c{text-align:right;color:var(--mu);font-family:ui-monospace,monospace}
 .row.done{border-color:#3d6b46} .row.part{border-color:#6b5a2c}
 .seg{display:grid;grid-template-columns:52px 1fr 260px;gap:14px;align-items:center;
   background:var(--sf);border:1px solid var(--bd);border-radius:10px;padding:12px 14px;margin-bottom:8px}
 .seg.has{border-color:#3d6b46} .seg.rec{border-color:var(--rd);background:#2a1f1f}
 .idx{font-family:ui-monospace,monospace;color:var(--mu)}
 .txt{font-size:17px} .ctl{display:flex;gap:8px;align-items:center;justify-content:flex-end}
 button{background:#333;color:var(--tx);border:1px solid var(--bd);border-radius:8px;
   padding:8px 14px;font-size:15px;cursor:pointer} button:hover{background:#3d3d3d}
 button.rec{background:#5a2a2a;border-color:#8a4444} audio{height:34px}
 .ok{color:var(--gd)}
</style></head><body>__BODY__ __EXTRA__</body></html>"""

_JS = r"""
const $=(s,r=document)=>r.querySelector(s);
const box=document.getElementById('lines');
let media=null, rec=null, chunks=[], active=null;

function micError(e){
  const insecure = !window.isSecureContext || !navigator.mediaDevices;
  const host = location.hostname;
  let msg;
  if(insecure){
    msg = '<b>The microphone is blocked on this address.</b><br>'+
      'Browsers only allow recording on a secure origin. <code>http://'+location.host+'</code> '+
      'is plain HTTP, so <code>navigator.mediaDevices</code> does not exist here.'+
      '<p style="margin:10px 0 4px"><b>Use the HTTPS port &mdash; it is 9443, not 9090:</b></p>'+
      '<p style="margin:0 0 8px"><a href="https://'+host+':9443'+location.pathname+'" '+
        'style="font-size:17px">https://'+host+':9443'+location.pathname+'</a></p>'+
      '<p style="margin:0 0 8px;color:#a3a3a3">Chrome warns about the self-signed certificate: '+
      '<b>Advanced &rarr; Proceed</b>. You accept it once.</p>'+
      '<details><summary style="cursor:pointer;color:#a0caff">other ways</summary>'+
      '<ol style="margin:8px 0 0 18px">'+
      '<li>On the box itself: <a href="http://localhost:9090'+location.pathname+'">'+
        'http://localhost:9090'+location.pathname+'</a></li>'+
      '<li>Tunnel: <code>ssh -L 9443:localhost:9443 '+host+'</code></li>'+
      '<li>Chrome flag: <code>chrome://flags/#unsafely-treat-insecure-origin-as-secure</code></li>'+
      '</ol></details>';
  } else if(e && e.name === 'NotAllowedError'){
    msg = '<b>Microphone permission was denied.</b><br>Click the mic icon in the address bar and allow it, then reload.';
  } else if(e && e.name === 'NotFoundError'){
    msg = '<b>No microphone found.</b><br>Plug one in or pick an input device in your OS settings.';
  } else {
    msg = '<b>Could not start the microphone.</b><br>'+(e ? (e.name+': '+e.message) : 'unknown error');
  }
  let el=document.getElementById('micerr');
  if(!el){ el=document.createElement('div'); el.id='micerr'; el.className='hint';
           el.style.borderColor='#8a4444'; el.style.background='#2a1f1f';
           document.querySelector('h1').insertAdjacentElement('afterend', el); }
  el.innerHTML = msg;
  el.scrollIntoView({block:'center',behavior:'smooth'});
}
async function getMic(){
  if(!navigator.mediaDevices || !window.isSecureContext){ micError(null); throw new Error('insecure'); }
  try{
    return await navigator.mediaDevices.getUserMedia(
      {audio:{echoCancellation:false,noiseSuppression:true,channelCount:1}});
  }catch(e){ micError(e); throw e; }
}
// warn BEFORE the user clicks a dead button
document.addEventListener('DOMContentLoaded',()=>{
  if(!window.isSecureContext || !navigator.mediaDevices) micError(null);
});


function wavFromBuffer(buf){                      // AudioBuffer -> 16-bit PCM WAV (mono)
  const sr=buf.sampleRate, ch=buf.numberOfChannels, n=buf.length;
  const mix=new Float32Array(n);
  for(let c=0;c<ch;c++){const d=buf.getChannelData(c); for(let i=0;i<n;i++) mix[i]+=d[i]/ch;}
  const ab=new ArrayBuffer(44+n*2), v=new DataView(ab);
  const wr=(o,s)=>{for(let i=0;i<s.length;i++)v.setUint8(o+i,s.charCodeAt(i));};
  wr(0,'RIFF'); v.setUint32(4,36+n*2,true); wr(8,'WAVEfmt '); v.setUint32(16,16,true);
  v.setUint16(20,1,true); v.setUint16(22,1,true); v.setUint32(24,sr,true);
  v.setUint32(28,sr*2,true); v.setUint16(32,2,true); v.setUint16(34,16,true);
  wr(36,'data'); v.setUint32(40,n*2,true);
  let o=44; for(let i=0;i<n;i++){const s=Math.max(-1,Math.min(1,mix[i])); v.setInt16(o,s<0?s*32768:s*32767,true); o+=2;}
  return new Blob([ab],{type:'audio/wav'});
}

async function mic(){
  if(!media) media=await getMic();
  return media;
}

async function start(i,el){
  if(active) return;
  let st;
  try{ st=await mic(); }catch(e){ return; }          // micError already explained it
  active=i; el.classList.add('rec'); chunks=[];
  rec=new MediaRecorder(st); rec.ondataavailable=e=>chunks.push(e.data);
  rec.start();
}

async function stop(i,el){
  if(active!==i||!rec) return;
  const done=new Promise(r=>rec.onstop=r); rec.stop(); await done;
  el.classList.remove('rec'); active=null;
  const blob=new Blob(chunks,{type:chunks[0]?.type||'audio/webm'});
  const ac=new AudioContext();
  const buf=await ac.decodeAudioData(await blob.arrayBuffer());
  const wav=wavFromBuffer(buf);
  const r=await fetch(`/api/narration?playlist=${PL}&video=${VID}&index=${i}`,
                      {method:'POST',headers:{'Content-Type':'audio/wav'},body:wav});
  const j=await r.json();
  if(!j.ok){alert('save failed: '+(j.error||'?')); return;}
  $('#done').textContent=j.have;
  el.classList.add('has');
  const a=$('audio',el)||document.createElement('audio');
  a.controls=true; a.src=URL.createObjectURL(wav);
  if(!$('audio',el)) $('.ctl',el).appendChild(a);
  if(j.complete) $('.sub').insertAdjacentHTML('beforeend',' <b class="ok">— complete, spec switched to your voice</b>');
}

LINES.forEach(l=>{
  const el=document.createElement('div');
  el.className='seg'+(HAVE.includes(l.i)?' has':'');
  el.innerHTML=`<div class="idx">${String(l.i).padStart(3,'0')}</div>
                <div class="txt">${l.text.replace(/</g,'&lt;')}</div>
                <div class="ctl"><button>hold to record</button></div>`;
  const b=$('button',el);
  b.onmousedown=()=>start(l.i,el); b.onmouseup=()=>stop(l.i,el); b.onmouseleave=()=>{if(active===l.i)stop(l.i,el);};
  b.ontouchstart=e=>{e.preventDefault();start(l.i,el);}; b.ontouchend=e=>{e.preventDefault();stop(l.i,el);};
  if(HAVE.includes(l.i)){const a=document.createElement('audio');a.controls=true;
    a.src=`/${PL}/${VID}/narration/${String(l.i).padStart(3,'0')}.wav`;$('.ctl',el).appendChild(a);}
  box.appendChild(el);
});

// Space records whichever line is focused/hovered next in sequence
let spaceIdx=null;
document.addEventListener('keydown',e=>{
  if(e.code!=='Space'||e.repeat||active) return;
  e.preventDefault();
  const el=[...document.querySelectorAll('.seg')].find(x=>!x.classList.contains('has'));
  if(!el) return;
  spaceIdx=LINES[[...document.querySelectorAll('.seg')].indexOf(el)].i;
  el.scrollIntoView({block:'center',behavior:'smooth'}); start(spaceIdx,el);
});
document.addEventListener('keyup',e=>{
  if(e.code!=='Space'||spaceIdx===null) return;
  e.preventDefault();
  const el=[...document.querySelectorAll('.seg')][spaceIdx-1];
  stop(spaceIdx,el); spaceIdx=null;
});
"""


# ---------------------------------------------------------------------------- /me : voice reference
# One passage, read once, becomes the narrator for all 81 videos. The text is assembled from REAL
# lines across the curriculum, so the reference carries the prosody of the thing it will actually
# be cloned for — reading a generic pangram gives a reference that sounds nothing like tutorial
# narration. Target ~2 minutes; the installer keeps the best 30s window.

# A short, phonetically varied read for zero-shot cloning. Chatterbox needs only ~10-30s, so
# asking for six minutes is a tax on the one person who has to read it. These lines are written to
# cover the vowel space and the awkward consonant clusters (th-, str-, -sps, -cts, plosives and
# sibilants), stay in the tutorial register, and end on complete words — a clip cut mid-word
# teaches the model a truncated phoneme.
QUICK_SCRIPT = [
    "Let's start with the basics, then build up to the tricky parts step by step.",
    "This function accepts a string, checks its length, and returns the third character.",
    "Python's zip pairs each name with its age, joining both lists in a single pass.",
    "Watch what happens when the loop finishes: the counter keeps its final value.",
    "Objects expose methods; modules expose functions; both just group behaviour together.",
    "Quality matters more than quantity, so choose clear examples over clever ones.",
    "Roughly ninety-five percent of bugs come from assumptions nobody wrote down.",
    "Thanks for watching — try it yourself, and tell me which part felt hardest.",
]


def reference_script(target_words=900):
    """Sentences drawn from across the curriculum: varied topics, natural technical delivery.

    Longer is better ONLY for window selection — the clone itself consumes ~30s. A wider spread
    also gets more phoneme coverage: symbols read aloud ("colon equals", "double underscore"),
    numbers, and library names all appear, which a generic passage would miss.
    """
    buckets, seen = [], set()
    for pl in PLAYLISTS:
        base = YT / pl
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            if not (d.is_dir() and (d / "transcript.json").exists()):
                continue
            segs = json.loads((d / "transcript.json").read_text())
            lines = []
            for sg in segs[1:-1]:                      # skip hook line and CTA tail
                t = " ".join(TAG.sub(" ", sg["text"]).split())
                if 10 <= len(t.split()) <= 28 and t not in seen:
                    seen.add(t)
                    lines.append(t)
            if lines:
                buckets.append(lines)
    # round-robin across videos so the passage never dwells on one topic
    out, n, depth = [], 0, 0
    while n < target_words and any(len(b) > depth for b in buckets):
        for b in buckets:
            if depth < len(b):
                out.append(b[depth])
                n += len(b[depth].split())
                if n >= target_words:
                    break
        depth += 1
    return out


def save_voice_ref(wav_bytes, ref_text=""):
    """Store the raw take, then run it through the same prepare/inspect path as a supplied file."""
    import sys as _s
    _s.path.insert(0, str(YT / "tools"))
    import set_voice
    raw = YT / "common" / "voice" / "_me_raw.wav"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if wav_bytes[:4] != b"RIFF":
        raise ValueError("not a WAV upload")
    raw.write_bytes(wav_bytes)
    if ref_text:
        # the exact words spoken: some cloners (e.g. OmniVoice) map phonemes far better when given
        # the reference transcript, and it documents what the reference actually contains
        (YT / "common" / "voice" / "my_voice_ref.txt").write_text(ref_text.strip() + "\n")
    prepared = set_voice.prepare(raw)
    problems, notes = set_voice.inspect(prepared)
    return {"ok": not problems, "notes": notes, "problems": problems,
            "prepared": str(prepared)}


def install_voice_ref():
    import sys as _s
    _s.path.insert(0, str(YT / "tools"))
    import set_voice
    raw = YT / "common" / "voice" / "_me_raw.wav"
    if not raw.exists():
        raise ValueError("record a take first")
    rc = set_voice.install(raw)
    return {"ok": rc == 0}


def me_html(mode="quick"):
    """`quick` (~40s, phonetically varied) is the default: cloning consumes ~10-30s, so a long
    read buys nothing but fatigue. `full` (~6 min, real curriculum lines) remains available when
    you want a wider pool for the window picker to choose from."""
    full = mode == "full"
    lines = reference_script() if full else QUICK_SCRIPT
    words = sum(len(l.split()) for l in lines)
    secs = words / 2.5
    other = ("quick 40-second version", "/me") if full else ("longer 6-minute version", "/me?mode=full")
    body = f"""
      <a class="back" href="/record">&larr; narration booth</a>
      <h1>record your voice</h1>
      <p class="sub">Read this once. It becomes the narrator for all 81 videos.
         {len(lines)} lines &middot; {words} words &middot; about {secs:.0f} seconds.</p>
      <p class="hint">Quiet room, no background noise, steady distance from the mic.
         Read at a natural pace in the tone you want the tutorials to have.
         Finish each sentence &mdash; don't stop mid-word. Mistakes are fine: the installer keeps
         the cleanest stretch. &middot; <a href="{other[1]}">{other[0]}</a></p>
      <div class="ctl" style="justify-content:flex-start;margin-bottom:16px">
        <button id="go">start recording</button>
        <span id="tm" class="idx">0:00</span>
        <span id="st" class="sub" style="margin:0"></span>
      </div>
      <div id="rp"></div>
      <div id="script">{''.join(f'<p class="txt" style="margin:0 0 12px">{l}</p>' for l in lines)}</div>"""
    extra = ("<script>const REF_TEXT=" + json.dumps(" ".join(lines)) + ";</script>"
             "<script>" + _ME_JS + "</script>")
    return _PAGE.replace("__TITLE__", "record your voice").replace("__BODY__", body).replace("__EXTRA__", extra)


_ME_JS = r"""
const $=(s)=>document.querySelector(s);
let media,rec,chunks=[],t0,timer;

function micError(e){
  const insecure = !window.isSecureContext || !navigator.mediaDevices;
  const host = location.hostname;
  let msg;
  if(insecure){
    msg = '<b>The microphone is blocked on this address.</b><br>'+
      'Browsers only allow recording on a secure origin. <code>http://'+location.host+'</code> '+
      'is plain HTTP, so <code>navigator.mediaDevices</code> does not exist here.'+
      '<p style="margin:10px 0 4px"><b>Use the HTTPS port &mdash; it is 9443, not 9090:</b></p>'+
      '<p style="margin:0 0 8px"><a href="https://'+host+':9443'+location.pathname+'" '+
        'style="font-size:17px">https://'+host+':9443'+location.pathname+'</a></p>'+
      '<p style="margin:0 0 8px;color:#a3a3a3">Chrome warns about the self-signed certificate: '+
      '<b>Advanced &rarr; Proceed</b>. You accept it once.</p>'+
      '<details><summary style="cursor:pointer;color:#a0caff">other ways</summary>'+
      '<ol style="margin:8px 0 0 18px">'+
      '<li>On the box itself: <a href="http://localhost:9090'+location.pathname+'">'+
        'http://localhost:9090'+location.pathname+'</a></li>'+
      '<li>Tunnel: <code>ssh -L 9443:localhost:9443 '+host+'</code></li>'+
      '<li>Chrome flag: <code>chrome://flags/#unsafely-treat-insecure-origin-as-secure</code></li>'+
      '</ol></details>';
  } else if(e && e.name === 'NotAllowedError'){
    msg = '<b>Microphone permission was denied.</b><br>Click the mic icon in the address bar and allow it, then reload.';
  } else if(e && e.name === 'NotFoundError'){
    msg = '<b>No microphone found.</b><br>Plug one in or pick an input device in your OS settings.';
  } else {
    msg = '<b>Could not start the microphone.</b><br>'+(e ? (e.name+': '+e.message) : 'unknown error');
  }
  let el=document.getElementById('micerr');
  if(!el){ el=document.createElement('div'); el.id='micerr'; el.className='hint';
           el.style.borderColor='#8a4444'; el.style.background='#2a1f1f';
           document.querySelector('h1').insertAdjacentElement('afterend', el); }
  el.innerHTML = msg;
  el.scrollIntoView({block:'center',behavior:'smooth'});
}
async function getMic(){
  if(!navigator.mediaDevices || !window.isSecureContext){ micError(null); throw new Error('insecure'); }
  try{
    return await navigator.mediaDevices.getUserMedia(
      {audio:{echoCancellation:false,noiseSuppression:true,channelCount:1}});
  }catch(e){ micError(e); throw e; }
}
// warn BEFORE the user clicks a dead button
document.addEventListener('DOMContentLoaded',()=>{
  if(!window.isSecureContext || !navigator.mediaDevices) micError(null);
});

function wavFromBuffer(buf){
  const sr=buf.sampleRate,ch=buf.numberOfChannels,n=buf.length,mix=new Float32Array(n);
  for(let c=0;c<ch;c++){const d=buf.getChannelData(c);for(let i=0;i<n;i++)mix[i]+=d[i]/ch;}
  const ab=new ArrayBuffer(44+n*2),v=new DataView(ab);
  const wr=(o,s)=>{for(let i=0;i<s.length;i++)v.setUint8(o+i,s.charCodeAt(i));};
  wr(0,'RIFF');v.setUint32(4,36+n*2,true);wr(8,'WAVEfmt ');v.setUint32(16,16,true);
  v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,sr,true);
  v.setUint32(28,sr*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);
  wr(36,'data');v.setUint32(40,n*2,true);
  let o=44;for(let i=0;i<n;i++){const s=Math.max(-1,Math.min(1,mix[i]));v.setInt16(o,s<0?s*32768:s*32767,true);o+=2;}
  return new Blob([ab],{type:'audio/wav'});
}
$('#go').onclick=async()=>{
  if(rec&&rec.state==='recording'){
    const done=new Promise(r=>rec.onstop=r);rec.stop();await done;clearInterval(timer);
    $('#go').textContent='start recording';$('#st').textContent='processing…';
    const blob=new Blob(chunks,{type:chunks[0]?.type||'audio/webm'});
    const ac=new AudioContext();const buf=await ac.decodeAudioData(await blob.arrayBuffer());
    const wav=wavFromBuffer(buf);
    $('#rp').innerHTML='';const a=document.createElement('audio');a.controls=true;
    a.src=URL.createObjectURL(wav);$('#rp').appendChild(a);
    const r=await fetch('/api/voiceref?ref_text='+encodeURIComponent(REF_TEXT),
                        {method:'POST',headers:{'Content-Type':'audio/wav'},body:wav});
    const j=await r.json();
    let h='<div class="hint" style="margin-top:12px">'+j.notes.map(n=>'• '+n).join('<br>');
    if(j.problems.length) h+='<br><b style="color:#ffaeaa">'+j.problems.join('<br>')+'</b>';
    h+='</div>';
    if(j.ok) h+='<button id="use">use this voice for all 81 videos</button>';
    $('#rp').insertAdjacentHTML('beforeend',h);$('#st').textContent='';
    if(j.ok) $('#use').onclick=async()=>{
      const x=await(await fetch('/api/voiceref?install=1',{method:'POST'})).json();
      $('#use').outerHTML=x.ok?'<b class="ok">installed — rebuild to hear it</b>':'<b>install failed</b>';
    };
    return;
  }
  try{ media=media||await getMic(); }catch(e){ return; }
  chunks=[];rec=new MediaRecorder(media);rec.ondataavailable=e=>chunks.push(e.data);rec.start();
  t0=Date.now();$('#go').textContent='stop';$('#st').textContent='recording…';
  timer=setInterval(()=>{const s=(Date.now()-t0)/1000;
    $('#tm').textContent=Math.floor(s/60)+':'+String(Math.floor(s%60)).padStart(2,'0');},250);
};
"""


def hub_status():
    """Voice + narration state for the hub header, so the recording routes are discoverable
    from the main page instead of being URLs you have to remember."""
    ref = YT / "common" / "voice" / "my_voice_ref.wav"
    pointed = total = done = partial = clips_have = clips_want = 0
    for pl in PLAYLISTS:
        base = YT / pl
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            if not (d.is_dir() and (d / "transcript.json").exists()):
                continue
            total += 1
            spec = json.loads((d / "spec.json").read_text())
            if str(spec.get("voice_ref", "")).endswith("my_voice_ref.wav"):
                pointed += 1
            have, want = progress(d)
            clips_have += have
            clips_want += want
            if want and have == want:
                done += 1
            elif have:
                partial += 1
    return {"cloned": ref.exists(), "pointed": pointed, "total": total,
            "read_done": done, "read_partial": partial,
            "clips_have": clips_have, "clips_want": clips_want}


def hub_header_html():
    s = hub_status()
    if s["cloned"]:
        vlabel, vcls = f"cloned · {s['pointed']}/{s['total']} videos", "ok"
    else:
        vlabel, vcls = "not recorded yet", "todo"
    pct = int(100 * s["clips_have"] / s["clips_want"]) if s["clips_want"] else 0
    if s["read_done"] == s["total"] and s["total"]:
        rlabel, rcls = "all read", "ok"
    elif s["clips_have"]:
        rlabel, rcls = f"{s['read_done']} done · {pct}% of clips", "part"
    else:
        rlabel, rcls = "not started", "todo"
    return (
        '<a class="vnav" href="/me" title="Read one passage; it becomes the narrator for every video">'
        f'<b>&#127908; My voice</b><span class="vb {vcls}">{vlabel}</span></a>'
        '<a class="vnav" href="/record" title="Read every line yourself — removes synthetic narration entirely">'
        f'<b>&#9210; Narration booth</b><span class="vb {rcls}">{rlabel}</span></a>')


HUB_CSS = """
.vnav{display:flex;flex-direction:column;gap:2px;text-decoration:none;color:var(--text);
  background:#2e2e2e;border:1px solid var(--border);border-radius:9px;padding:6px 12px;margin-left:8px}
.vnav:hover{background:#383838;border-color:#5a5a5a}
.vnav b{font-size:12px;font-weight:700;line-height:1.2}
.vb{font-size:10.5px;line-height:1.2}
.vb.ok{color:#a6d2a2} .vb.part{color:#eec12f} .vb.todo{color:#a3a3a3}
"""
