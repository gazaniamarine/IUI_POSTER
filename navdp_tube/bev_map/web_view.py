"""Tiny browser front-end for headless machines (e.g. working over SSH).

Streams the dashboard as MJPEG and forwards key presses from the page. Standard library only.
Open http://localhost:<port> - VS Code Remote-SSH forwards the port automatically
(or: ssh -L 8090:localhost:8090 <server>).
"""

import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>BEV map</title>
<style>
 body{margin:0;background:#1e1e1e;color:#ddd;font:14px system-ui,sans-serif;padding:12px 16px}
 img{max-width:100%;border:1px solid #444;display:block}
 .keys{margin-top:10px;display:flex;flex-wrap:wrap;gap:6px 18px}
 kbd{background:#333;border:1px solid #555;border-radius:4px;padding:1px 6px}
 .btns button{margin:4px 4px 0 0;padding:6px 12px;background:#333;color:#ddd;border:1px solid #555;border-radius:4px;cursor:pointer}
 #st{color:#8c8}
</style></head><body>
<img id="v" src="/stream" alt="waiting for frames...">
<div class="keys">
 <span><kbd>&uarr;&darr;&larr;&rarr;</kbd>/<kbd>WASD</kbd> drive (hold)</span>
 <span><kbd>J</kbd><kbd>L</kbd> pan <kbd>I</kbd><kbd>K</kbd> tilt</span>
 <span><kbd>G</kbd> true outline</span><span><kbd>R</kbd> clear map</span>
 <span><kbd>P</kbd> save</span><span><kbd>Q</kbd> quit</span>
 <span>click camera/map: set goal (navigation)</span><span><kbd>X</kbd> cancel goal</span><span id="st"></span>
</div>
<div class="btns">
 <button data-k="g">Toggle true outline</button><button data-k="r">Clear map</button>
 <button data-k="p">Save map</button><button data-k="q">Quit</button>
</div>
<script>
const DRIVE={ArrowUp:'up',ArrowDown:'down',ArrowLeft:'left',ArrowRight:'right',w:'up',s:'down',a:'left',d:'right'};
const held=new Set();
function send(k){fetch('/key?k='+encodeURIComponent(k)).then(r=>r.text()).then(t=>{if(t)document.getElementById('st').textContent=t;}).catch(()=>{});}
document.addEventListener('keydown',e=>{
  const k=e.key.length==1?e.key.toLowerCase():e.key;
  if(DRIVE[k]){e.preventDefault(); if(!held.has(k)){held.add(k); send(DRIVE[k]);} return;}
  if(k==' '){e.preventDefault(); send('stop'); return;}
  if(!e.repeat && 'jlikgrpqx'.includes(k)) send(k);
});
document.addEventListener('keyup',e=>{
  const k=e.key.length==1?e.key.toLowerCase():e.key;
  if(DRIVE[k]){held.delete(k); if(held.size==0) send('stop'); else send(DRIVE[[...held].pop()]);}
});
window.addEventListener('blur',()=>{held.clear(); send('stop');});
// keep re-sending the held drive key: the real rover stops by itself if these stop arriving (dead-man)
setInterval(()=>{if(held.size) send(DRIVE[[...held].pop()]);},200);
// click on the dashboard: position as a fraction of the image (the page may scale it)
document.getElementById('v').addEventListener('click',e=>{const r=e.target.getBoundingClientRect();
  send('click:'+((e.clientX-r.left)/r.width).toFixed(4)+','+((e.clientY-r.top)/r.height).toFixed(4));});
document.querySelectorAll('button[data-k]').forEach(b=>b.onclick=()=>send(b.dataset.k));
</script></body></html>"""


class WebView:
    def __init__(self, host="127.0.0.1", port=8090, quality=80):
        self.quality = quality
        self.keys = queue.Queue()
        self._jpg, self._seq = None, 0
        self._cond = threading.Condition()
        self.status = ""
        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                url = urlparse(self.path)
                if url.path == "/":
                    body = PAGE.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif url.path == "/key":
                    k = parse_qs(url.query).get("k", [""])[0]
                    if k:
                        view.keys.put(k)
                    body = view.status.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif url.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    seen = -1
                    try:
                        while True:
                            with view._cond:
                                view._cond.wait_for(lambda: view._seq != seen, timeout=5.0)
                                jpg, seen = view._jpg, view._seq
                            if jpg is None:
                                continue
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                             + f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.send_error(404)

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://localhost:{port}"

    def show(self, bgr):
        ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            with self._cond:
                self._jpg, self._seq = enc.tobytes(), self._seq + 1
                self._cond.notify_all()

    def pop_keys(self):
        out = []
        while not self.keys.empty():
            out.append(self.keys.get_nowait())
        return out

    def close(self):
        self.server.shutdown()


class Display:
    """cv2 window when a screen is available, otherwise the browser page. keys() returns key names
    like the page sends them ('up', 'down', 'left', 'right', 'stop' or a single letter), and
    'click:<fx>,<fy>' for a left click, as a fraction of the image width / height."""

    ARROWS = {65361: "left", 65362: "up", 65363: "right", 65364: "down",
              2424832: "left", 2490368: "up", 2555904: "right", 2621440: "down",
              16777234: "left", 16777235: "up", 16777236: "right", 16777237: "down"}
    WASD = {"w": "up", "s": "down", "a": "left", "d": "right", " ": "stop"}

    def __init__(self, title="BEV", web=False, host="127.0.0.1", port=8090):
        self.title = title
        self.web = WebView(host, port) if web else None
        self._pending = []
        self._shape = None
        self._mouse = False

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self._shape is not None:
            self._pending.append(f"click:{x / self._shape[1]:.4f},{y / self._shape[0]:.4f}")

    @property
    def url(self):
        return self.web.url if self.web else None

    def show(self, bgr, status=""):
        if self.web is not None:
            self.web.status = status
            self.web.show(bgr)
        else:
            cv2.imshow(self.title, bgr)
            self._shape = bgr.shape[:2]
            if not self._mouse:
                cv2.setMouseCallback(self.title, self._on_mouse)
                self._mouse = True
            k = cv2.waitKeyEx(1)
            if k >= 0:
                if k in self.ARROWS:
                    self._pending.append(self.ARROWS[k])
                elif k == 27:
                    self._pending.append("q")
                elif k < 256:
                    c = chr(k & 0xFF).lower()
                    self._pending.append(self.WASD.get(c, c))

    def keys(self):
        if self.web is not None:
            return self.web.pop_keys()
        out, self._pending = self._pending, []
        return out

    def close(self):
        if self.web is not None:
            self.web.close()
        else:
            cv2.destroyAllWindows()
