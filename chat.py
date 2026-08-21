#!/usr/bin/env python3
"""Local chat UI + OpenAI-compatible API for the Qwen3.5 Defiant models.

Serves:
  GET  /                     chat UI
  GET  /v1/models            OpenAI model list
  POST /v1/chat/completions  OpenAI chat completions (stream + non-stream)
  POST /ui/chat              internal streaming endpoint for the UI

Everything proxies to Ollama, which must already be running (./run-qwen35.sh start).
Stdlib only - no pip installs.
"""
import json, time, uuid, urllib.request, urllib.error, argparse, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OLLAMA = "http://127.0.0.1:11435"

# Friendly alias -> real ollama tag. Aliases keep the UI and API stable
# even if the underlying tags get re-imported under different names.
MODELS = {
    "qwen35-fast":   {"tag": "qwen3_5_9b_iq3:latest",      "ctx": 65536, "vision": False},
    "qwen35-quality":{"tag": "qwen35-defiant-q4km:latest", "ctx": 16384, "vision": False},
    "qwen35-vision": {"tag": "qwen35-vision:latest",       "ctx": 16384, "vision": True},
}
DEFAULT_MODEL = "qwen35-fast"


def resolve(name):
    """Accept an alias or a raw ollama tag."""
    if name in MODELS:
        return MODELS[name]
    for alias, spec in MODELS.items():
        if spec["tag"] == name or spec["tag"].split(":")[0] == name:
            return spec
    return {"tag": name, "ctx": 8192, "vision": False}


def ollama_chat(payload, stream):
    req = urllib.request.Request(
        f"{OLLAMA}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=900)


def build_payload(model_spec, messages, opts, think, fmt, stream):
    payload = {
        "model": model_spec["tag"],
        "messages": messages,
        "stream": stream,
        "think": think,
        "options": opts,
    }
    if fmt is not None:
        payload["format"] = fmt
    return payload


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):  # keep the console readable
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % a))

    # ---------- helpers ----------
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _stream_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # Must be "close": an SSE body carries no Content-Length, so with
        # keep-alive the browser never observes EOF and the client-side
        # reader stalls forever after the final event.
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.close_connection = True

    def _sse(self, obj):
        data = obj if isinstance(obj, str) else json.dumps(obj)
        self.wfile.write(f"data: {data}\n\n".encode())
        self.wfile.flush()

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    # ---------- routes ----------
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, HTML, "text/html; charset=utf-8")
        if self.path == "/v1/models":
            now = int(time.time())
            return self._send(200, {
                "object": "list",
                "data": [
                    {"id": a, "object": "model", "created": now, "owned_by": "local",
                     "context_length": s["ctx"], "vision": s["vision"]}
                    for a, s in MODELS.items()
                ],
            })
        if self.path == "/healthz":
            try:
                urllib.request.urlopen(f"{OLLAMA}/api/version", timeout=3).read()
                return self._send(200, {"ok": True, "ollama": OLLAMA})
            except Exception as e:
                return self._send(503, {"ok": False, "error": str(e)})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            if self.path == "/v1/chat/completions":
                return self.openai_completions()
            if self.path == "/ui/chat":
                return self.ui_chat()
            self._send(404, {"error": "not found"})
        except urllib.error.HTTPError as e:
            self._send(502, {"error": {"message": e.read().decode()[:500], "type": "upstream"}})
        except BrokenPipeError:
            pass
        except Exception as e:
            try:
                self._send(500, {"error": {"message": str(e), "type": "server_error"}})
            except Exception:
                pass

    # ---------- OpenAI-compatible ----------
    def openai_completions(self):
        b = self._body()
        spec = resolve(b.get("model", DEFAULT_MODEL))
        stream = bool(b.get("stream", False))

        opts = {"num_ctx": int(b.get("max_context", spec["ctx"]))}
        if b.get("temperature") is not None: opts["temperature"] = b["temperature"]
        if b.get("top_p") is not None:       opts["top_p"] = b["top_p"]
        if b.get("seed") is not None:        opts["seed"] = b["seed"]
        if b.get("stop"): opts["stop"] = b["stop"] if isinstance(b["stop"], list) else [b["stop"]]

        # response_format -> ollama `format` (json mode / structured outputs)
        fmt = None
        rf = b.get("response_format") or {}
        if rf.get("type") == "json_object":
            fmt = "json"
        elif rf.get("type") == "json_schema":
            fmt = (rf.get("json_schema") or {}).get("schema")

        think = b.get("reasoning") is not False and b.get("think", True)

        # OpenAI semantics: max_tokens bounds the *answer*. Ollama's num_predict
        # counts reasoning tokens too, so a thinking model will happily burn the
        # whole budget before emitting any content. Add headroom so max_tokens
        # means what an OpenAI client expects it to mean.
        mt = b.get("max_tokens") or b.get("max_completion_tokens")
        if mt:
            budget = int(b.get("reasoning_budget", 2048)) if think else 0
            opts["num_predict"] = int(mt) + budget

        msgs = []
        for m in b.get("messages", []):
            msgs.append(normalize_message(m))

        payload = build_payload(spec, msgs, opts, think, fmt, stream)
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        model_id = b.get("model", DEFAULT_MODEL)

        if not stream:
            raw = json.loads(ollama_chat(payload, False).read())
            msg = raw.get("message", {})
            out = {"role": "assistant", "content": msg.get("content", "")}
            if msg.get("thinking"):
                out["reasoning_content"] = msg["thinking"]
            pt = raw.get("prompt_eval_count", 0) or 0
            ct = raw.get("eval_count", 0) or 0
            return self._send(200, {
                "id": cid, "object": "chat.completion", "created": created, "model": model_id,
                "choices": [{"index": 0, "message": out,
                             "finish_reason": "length" if raw.get("done_reason") == "length" else "stop"}],
                "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
            })

        # streaming
        self._stream_start()
        resp = ollama_chat(payload, True)
        first = True
        for line in resp:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            msg = ev.get("message", {}) or {}
            delta = {}
            if first:
                delta["role"] = "assistant"
                first = False
            if msg.get("thinking"):
                delta["reasoning_content"] = msg["thinking"]
            if msg.get("content"):
                delta["content"] = msg["content"]
            if delta:
                self._sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                           "model": model_id,
                           "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
            if ev.get("done"):
                pt = ev.get("prompt_eval_count", 0) or 0
                ct = ev.get("eval_count", 0) or 0
                self._sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                           "model": model_id,
                           "choices": [{"index": 0, "delta": {},
                                        "finish_reason": "length" if ev.get("done_reason") == "length" else "stop"}],
                           "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                                     "total_tokens": pt + ct}})
                break
        self._sse("[DONE]")

    # ---------- UI ----------
    def ui_chat(self):
        b = self._body()
        spec = resolve(b.get("model", DEFAULT_MODEL))
        opts = {"num_ctx": int(b.get("num_ctx", spec["ctx"])),
                "temperature": b.get("temperature", 0.7)}
        if b.get("num_predict"): opts["num_predict"] = int(b["num_predict"])

        payload = build_payload(spec, b.get("messages", []), opts,
                                bool(b.get("think", True)), None, True)
        self._stream_start()
        t0 = time.time()
        resp = ollama_chat(payload, True)
        for line in resp:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            msg = ev.get("message", {}) or {}
            out = {}
            if msg.get("thinking"): out["thinking"] = msg["thinking"]
            if msg.get("content"):  out["content"] = msg["content"]
            if out:
                self._sse(out)
            if ev.get("done"):
                ct = ev.get("eval_count", 0) or 0
                ed = (ev.get("eval_duration", 0) or 0) / 1e9
                self._sse({"done": True, "tokens": ct,
                           "tps": round(ct / ed, 1) if ed else 0,
                           "elapsed": round(time.time() - t0, 1)})
                break
        self._sse("[DONE]")


def normalize_message(m):
    """OpenAI messages allow content parts; ollama wants text + images[]."""
    c = m.get("content")
    if isinstance(c, list):
        text, images = [], []
        for part in c:
            if part.get("type") == "text":
                text.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                if url.startswith("data:"):
                    url = url.split(",", 1)[-1]
                images.append(url)
        out = {"role": m.get("role", "user"), "content": "\n".join(text)}
        if images:
            out["images"] = images
        return out
    out = {"role": m.get("role", "user"), "content": c or ""}
    if m.get("images"):
        out["images"] = m["images"]
    return out


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Qwen3.5 Defiant</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0f1115;--panel:#161a21;--line:#252b36;--fg:#e6e9ef;--dim:#8b95a7;
--accent:#7aa2f7;--user:#1f2937;--think:#12151b}
*{box-sizing:border-box}
body{margin:0;font:15px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
background:var(--bg);color:var(--fg);height:100vh;display:flex;flex-direction:column}
header{display:flex;gap:10px;align-items:center;padding:10px 14px;border-bottom:1px solid var(--line);
background:var(--panel);flex-wrap:wrap}
header h1{font-size:15px;margin:0 8px 0 0;font-weight:600}
select,button,input{background:#1b2029;color:var(--fg);border:1px solid var(--line);
border-radius:8px;padding:6px 10px;font:inherit}
button{cursor:pointer}button:hover{border-color:var(--accent)}
#log{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:16px}
.msg{max-width:820px;width:100%;margin:0 auto}
.role{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);margin-bottom:5px}
.bubble{white-space:pre-wrap;word-wrap:break-word}
.user .bubble{background:var(--user);padding:10px 13px;border-radius:10px}
details.think{background:var(--think);border:1px solid var(--line);border-radius:8px;
margin-bottom:10px;font-size:13.5px;color:var(--dim)}
details.think summary{cursor:pointer;padding:7px 11px;user-select:none}
details.think .inner{padding:0 11px 10px;white-space:pre-wrap}
.meta{font-size:11.5px;color:var(--dim);margin-top:6px}
footer{border-top:1px solid var(--line);background:var(--panel);padding:12px 14px}
.row{max-width:820px;margin:0 auto;display:flex;gap:8px;align-items:flex-end}
textarea{flex:1;resize:none;background:#1b2029;color:var(--fg);border:1px solid var(--line);
border-radius:10px;padding:10px 12px;font:inherit;max-height:200px}
img.thumb{max-width:220px;border-radius:8px;margin-top:8px;display:block}
.pill{font-size:11.5px;color:var(--dim);border:1px solid var(--line);border-radius:999px;padding:3px 9px}
.err{color:#f7768e}
.notice{font-size:12.5px;color:#e0af68;border:1px solid #3d3420;background:#1e1a10;
border-radius:8px;padding:7px 11px;margin-bottom:9px}
.notice b{color:#f0c674}
</style></head><body>
<header>
  <h1>Qwen3.5 Defiant</h1>
  <select id="model"></select>
  <label class="pill">ctx <input id="ctx" type="number" style="width:82px;padding:2px 6px" value="16384"></label>
  <label class="pill"><input id="think" type="checkbox" checked> reasoning</label>
  <button id="pick">+ image</button>
  <span id="imgname" class="pill" style="display:none"></span>
  <button id="clear">clear</button>
  <span style="flex:1"></span>
  <span class="pill" id="status">ready</span>
  <input type="file" id="file" accept="image/*" style="display:none">
</header>
<div id="log"></div>
<footer><div class="row">
  <textarea id="input" rows="1" placeholder="Message  (Enter to send, Shift+Enter for newline)"></textarea>
  <button id="send">Send</button>
</div></footer>
<script>
const $=s=>document.querySelector(s);
const log=$('#log');
let history=JSON.parse(localStorage.getItem('qwen35-chat')||'[]');
let pendingImage=null, busy=false;
// Ollama pins context at runner startup, so changing model OR ctx forces a full
// reload (~25 s). Track what is loaded so we can warn instead of looking frozen.
let lastLoad={model:null,ctx:null};

async function loadModels(){
  const r=await fetch('/v1/models'); const d=await r.json();
  d.data.forEach(m=>{
    const o=document.createElement('option');
    o.value=m.id; o.textContent=m.id+(m.vision?'  (vision)':'')+'  · '+(m.context_length/1024)+'K';
    o.dataset.ctx=m.context_length; o.dataset.vision=m.vision;
    $('#model').appendChild(o);
  });
  syncCtx();
}
function syncCtx(){
  const o=$('#model').selectedOptions[0]; if(o) $('#ctx').value=o.dataset.ctx;
}
$('#model').onchange=syncCtx;

function esc(t){const d=document.createElement('div');d.textContent=t;return d.innerHTML}

function render(){
  log.innerHTML='';
  history.forEach(m=>addBubble(m.role,m.content,m.thinking,m.meta,m.image));
  log.scrollTop=log.scrollHeight;
}
function addBubble(role,content,thinking,meta,image){
  const w=document.createElement('div'); w.className='msg '+role;
  let h='<div class="role">'+role+'</div>';
  if(thinking) h+='<details class="think"><summary>reasoning</summary><div class="inner">'+esc(thinking)+'</div></details>';
  h+='<div class="bubble">'+esc(content||'')+'</div>';
  if(image) h+='<img class="thumb" src="data:image/png;base64,'+image+'">';
  if(meta) h+='<div class="meta">'+esc(meta)+'</div>';
  w.innerHTML=h; log.appendChild(w); return w;
}

$('#pick').onclick=()=>$('#file').click();
$('#file').onchange=e=>{
  const f=e.target.files[0]; if(!f) return;
  const rd=new FileReader();
  rd.onload=()=>{pendingImage=rd.result.split(',')[1];
    $('#imgname').style.display=''; $('#imgname').textContent=f.name;};
  rd.readAsDataURL(f);
};
$('#clear').onclick=()=>{history=[];save();render()};
function save(){localStorage.setItem('qwen35-chat',JSON.stringify(history))}

$('#input').addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}
});
$('#send').onclick=send;

async function send(){
  if(busy) return;
  const text=$('#input').value.trim();
  if(!text&&!pendingImage) return;
  const um={role:'user',content:text}; if(pendingImage) um.image=pendingImage;
  history.push(um); save();
  $('#input').value=''; render();

  busy=true;
  const model=$('#model').value, ctx=parseInt($('#ctx').value)||8192;
  const reload = lastLoad.model!==null && (lastLoad.model!==model || lastLoad.ctx!==ctx);
  const cold   = lastLoad.model===null;
  $('#status').textContent = (reload||cold) ? 'loading model…' : 'generating…';

  const wire=history.map(m=>{
    const o={role:m.role,content:m.content};
    if(m.image) o.images=[m.image];
    return o;
  });
  const holder=addBubble('assistant','',null,null,null);
  const bubble=holder.querySelector('.bubble');
  let det=null, thinkBuf='', contentBuf='';

  // Warn about the reload stall rather than letting the UI look hung.
  let notice=null;
  if(reload||cold){
    notice=document.createElement('div'); notice.className='notice';
    const why = reload
      ? (lastLoad.model!==model ? 'Model changed' : 'Context changed to '+(ctx/1024)+'K')
      : 'First message';
    notice.innerHTML='<b>'+why+'</b> — reloading into VRAM, this reply may take '
      +'<b>~25 s</b> to start. Later messages are instant.';
    holder.insertBefore(notice,bubble);
    log.scrollTop=log.scrollHeight;
  }
  const clearNotice=()=>{ if(notice){notice.remove(); notice=null;
    $('#status').textContent='generating…'; } };

  try{
    const res=await fetch('/ui/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model:model,messages:wire,
        num_ctx:ctx,think:$('#think').checked})});
    if(!res.ok) throw new Error('HTTP '+res.status+' '+(await res.text()).slice(0,200));
    const rd=res.body.getReader(), dec=new TextDecoder();
    let buf='', finished=false;
    while(!finished){
      const {value,done}=await rd.read(); if(done) break;
      buf+=dec.decode(value,{stream:true});
      const parts=buf.split('\n\n'); buf=parts.pop();
      for(const p of parts){
        const line=p.replace(/^data: /,'').trim();
        if(!line) continue;
        if(line==='[DONE]'){ finished=true; break; }
        const ev=JSON.parse(line);
        if(ev.thinking||ev.content){ clearNotice(); lastLoad={model:model,ctx:ctx}; }
        if(ev.thinking){
          thinkBuf+=ev.thinking;
          if(!det){det=document.createElement('details');det.className='think';
            det.innerHTML='<summary>reasoning</summary><div class="inner"></div>';
            det.open=true; holder.insertBefore(det,bubble);}
          det.querySelector('.inner').textContent=thinkBuf;
        }
        if(ev.content){contentBuf+=ev.content;bubble.textContent=contentBuf;}
        if(ev.done){
          if(det) det.open=false;
          const meta=ev.tokens+' tok · '+ev.tps+' tok/s · '+ev.elapsed+'s';
          const md=document.createElement('div'); md.className='meta'; md.textContent=meta;
          holder.appendChild(md);
          history.push({role:'assistant',content:contentBuf,thinking:thinkBuf,meta:meta});
          save();
        }
        log.scrollTop=log.scrollHeight;
      }
    }
  }catch(e){
    clearNotice(); lastLoad={model:null,ctx:null};
    bubble.classList.add('err'); bubble.textContent='Error: '+e.message;
  }finally{
    busy=false; $('#status').textContent='ready';
    pendingImage=null; $('#imgname').style.display='none'; $('#file').value='';
  }
}
loadModels().then(render);
</script></body></html>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8181)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ollama", default=OLLAMA)
    a = ap.parse_args()
    OLLAMA = a.ollama
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"chat UI      http://{a.host}:{a.port}/")
    print(f"OpenAI API   http://{a.host}:{a.port}/v1   (base_url for any OpenAI client)")
    print(f"upstream     {OLLAMA}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
