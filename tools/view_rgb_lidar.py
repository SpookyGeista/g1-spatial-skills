#!/usr/bin/env python3
"""Local, read-only browser preview. No GPU, XR, DDS or robot commands."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from punktnull.lidar_stereo import StereoWindow
from punktnull.rgb_lidar import read_packet

HTML = '''<!doctype html><html lang="de"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>G1 · RGB-LiDAR</title><style>
body{margin:0;background:#080d14;color:#e2e8f0;font:17px system-ui}
main{max-width:1280px;margin:4vh auto;padding:24px}h1{font-size:28px}
#status{padding:16px;background:#1c293a;border-radius:12px}
#source{color:#a8b8ce;overflow-wrap:anywhere}img{width:100%;border:1px solid #334155;border-radius:12px}
.eyes{display:flex;justify-content:space-around;color:#a8b8ce;margin:16px}
</style><main><h1>G1 · Farbige LiDAR-Punkte</h1>
<p id="status">Verbindung zur Vorschau …</p><p id="source"></p>
<div class="eyes"><span>Linkes Auge</span><span>Rechtes Auge</span></div>
<img id="cloud" alt="Stereo-Punktwolke" hidden>
<p>Die Farben stammen aus der simulierten Kamera. Pro Bild wird ein einzelner
LiDAR-Scan gezeigt: wenige Punkte, kein geschlossenes Kamerabild.</p>
<p>Diese PC-Vorschau benötigt keine VR-Brille und sendet keine Steuerbefehle.</p></main>
<script>
async function refresh(){
 const pic=document.getElementById('cloud'), status=document.getElementById('status');
 try{
  const response=await fetch('/frame',{cache:'no-store',signal:AbortSignal.timeout(3000)});
  if(!response.ok)throw Error('HTTP '+response.status);
  const frame=await response.json();status.textContent=frame.status;
  document.getElementById('source').textContent=frame.source;
  pic.hidden=!frame.image;if(frame.image)pic.src='data:image/png;base64,'+frame.image;
 }catch(e){status.textContent='Vorschau nicht erreichbar – Anzeige angehalten.';pic.hidden=true;}
 setTimeout(refresh,250);
}refresh();
</script></html>'''


def preview_frame(run, replay=False):
    result = dict(source=str(run), image=None)
    try:
        xyz, colors, meta = read_packet(run/'sim/rgb_lidar/latest.npz', replay=replay)
        left, right = StereoWindow().render(xyz, colors, [0, 0, 0], [1, 0, 0, 0])
        output = io.BytesIO()
        Image.fromarray(np.concatenate((left, right), axis=1)).save(output, format='PNG')
        result['image'] = base64.b64encode(output.getvalue()).decode('ascii')
        label = 'AUFZEICHNUNG · kein Livebild' if replay else 'LIVE'
        age = '' if replay else f" · Bildalter {time.monotonic()-meta['capture_monotonic']:.2f} s"
        result['status'] = f"{label} · {len(xyz)} farbige Punkte · Frame {meta.get('frame_id', '?')}{age}"
        result['frame_id'] = meta.get('frame_id')
    except FileNotFoundError:
        result['status'] = 'Warte auf Sensordaten. Simulation mit --rgb-lidar starten und READY abwarten.'
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        result['status'] = 'Keine frischen gültigen Sensordaten. Simulation angehalten oder beendet. '+str(exc)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--replay', action='store_true', help='Clearly labelled historical image; no live-data claim')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_dir():
        parser.error('Laufordner existiert nicht: '+str(run))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                data, kind = HTML.encode(), 'text/html; charset=utf-8'
            elif self.path == '/frame':
                data = json.dumps(preview_frame(run, args.replay)).encode()
                kind = 'application/json'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_args):
            pass

    with HTTPServer(('127.0.0.1', args.port), Handler) as server:
        print(f'PC-Vorschau: http://127.0.0.1:{server.server_port} – Strg+C beendet die Vorschau.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
