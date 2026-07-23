from __future__ import annotations

import base64
import json
from pathlib import Path


def frame_gallery_html(frames: list[Path], center_index: int) -> str:
    """Build a self-contained frame viewer with fullscreen navigation."""
    images = [f"data:image/jpeg;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}" for path in frames]
    offsets = [index - center_index for index in range(len(frames))]
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; height: 100%; background: #0e1117; color: white; font-family: sans-serif; }}
#viewer {{ position: relative; height: 690px; display: flex; flex-direction: column; background: #0e1117; }}
#stage {{ position: relative; flex: 1; min-height: 0; display: flex; align-items: center; justify-content: center; }}
#frame {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
.arrow {{ position: absolute; top: 50%; transform: translateY(-50%); width: 52px; height: 72px;
  border: 0; border-radius: 10px; background: rgba(0,0,0,.62); color: white; font-size: 38px; cursor: pointer; }}
.arrow:hover {{ background: rgba(255,75,75,.9); }}
#previous {{ left: 12px; }} #next {{ right: 12px; }}
#fullscreen {{ position: absolute; top: 12px; right: 12px; border: 0; border-radius: 8px;
  padding: 9px 13px; background: rgba(0,0,0,.68); color: white; cursor: pointer; font-size: 16px; }}
#caption {{ text-align: center; padding: 9px; font-size: 16px; }}
#thumbs {{ height: 92px; display: flex; gap: 6px; overflow-x: auto; padding: 6px; background: #161b22; }}
.thumb {{ height: 76px; opacity: .62; cursor: pointer; border: 2px solid transparent; }}
.thumb.active {{ opacity: 1; border-color: #ff4b4b; }}
#viewer:fullscreen {{ width: 100vw; height: 100vh; }}
#viewer:fullscreen #stage {{ background: black; }}
</style></head><body>
<div id="viewer">
  <div id="stage">
    <img id="frame" alt="Candidate frame">
    <button id="previous" class="arrow" aria-label="Previous frame">‹</button>
    <button id="next" class="arrow" aria-label="Next frame">›</button>
    <button id="fullscreen">⛶ Fullscreen</button>
  </div>
  <div id="caption"></div><div id="thumbs"></div>
</div>
<script>
const images = {json.dumps(images)};
const offsets = {json.dumps(offsets)};
// Always begin at the first frame so review naturally moves forward in time.
let current = 0;
const viewer = document.getElementById('viewer');
const image = document.getElementById('frame');
const caption = document.getElementById('caption');
const thumbs = document.getElementById('thumbs');
images.forEach((source, index) => {{
  const thumb = document.createElement('img'); thumb.src = source; thumb.className = 'thumb';
  thumb.onclick = () => show(index); thumbs.appendChild(thumb);
}});
function show(index) {{
  if (!images.length) return;
  current = (index + images.length) % images.length; image.src = images[current];
  const offset = offsets[current];
  caption.textContent = `Frame ${{current + 1}} of ${{images.length}} · ${{offset >= 0 ? '+' : ''}}${{offset}} from detected contact`;
  [...thumbs.children].forEach((item, i) => item.classList.toggle('active', i === current));
  thumbs.children[current]?.scrollIntoView({{behavior: 'smooth', block: 'nearest', inline: 'center'}});
}}
document.getElementById('previous').onclick = () => show(current - 1);
document.getElementById('next').onclick = () => show(current + 1);
document.getElementById('fullscreen').onclick = async () => {{
  if (document.fullscreenElement) await document.exitFullscreen(); else await viewer.requestFullscreen();
}};
document.addEventListener('keydown', event => {{
  if (event.key === 'ArrowLeft') show(current - 1);
  if (event.key === 'ArrowRight') show(current + 1);
  if (event.key === 'Escape' && document.fullscreenElement) document.exitFullscreen();
}});
show(current);
</script></body></html>"""
