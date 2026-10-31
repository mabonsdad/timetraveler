// Historic Soundscapes player: fixed point of view, layered equirect panorama.
//
// Every layer from the pipeline is drawn as a piece of sphere covering exactly its bounding box,
// in scene.json order (far -> near). With a fixed viewpoint this is the same as compositing
// the layers in 2D, but it keeps them as separate objects we can fade, restyle and highlight.
//
// URL parameters (handy for stage presets and capture):
//   scene=path/to/scene.json  era=1926  style=0..1  dawn=0..1  mode=look|flat360|flat180
//   yaw=deg  pitch=deg  fov=deg  ui=0
import * as THREE from 'three';

const params = new URLSearchParams(location.search);
const sceneUrl = new URL(params.get('scene') || '../projects/test_lane/scene.json', location.href);
const R = 500;
const DEG = Math.PI / 180;

// ---------------------------------------------------------------- renderer + cameras
const canvas = document.getElementById('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const lookScene = new THREE.Scene();
const flatScene = new THREE.Scene();
const lookCam = new THREE.PerspectiveCamera(Number(params.get('fov') || 75), 1, 1, 2000);
const flatCam = new THREE.OrthographicCamera(-1, 1, 0.5, -0.5, -10, 10);
const view = { yaw: Number(params.get('yaw') || 0), pitch: Number(params.get('pitch') || 0) };
const MODES = ['look', 'flat360', 'flat180'];
let mode = MODES.includes(params.get('mode')) ? params.get('mode') : 'look';

// Direction for (yaw, pitch) in degrees, matching the pipeline's equirect convention
// (image centre = yaw 0, yaw increases to the right) and three's SphereGeometry mapping.
function dir(yawDeg, pitchDeg) {
  const lon = yawDeg * DEG, lat = pitchDeg * DEG;
  return new THREE.Vector3(-Math.cos(lon) * Math.cos(lat), Math.sin(lat), -Math.sin(lon) * Math.cos(lat));
}

// ---------------------------------------------------------------- shared shader state
const U = {
  sepia: { value: 0 }, sat: { value: 1 }, tint: { value: new THREE.Vector3(1, 1, 1) },
  dawn: { value: Number(params.get('dawn') ?? 1) }, time: { value: 0 },
  styleFrom: { value: 0 }, styleTo: { value: 0 }, sweep: { value: 1 },
};

const VERT = /* glsl */`
varying vec2 vUv;
void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;

// Textures are sampled raw (no colour-space conversion) so the grading maths matches the
// Python renderer, which works on sRGB values directly.
const FRAG = /* glsl */`
uniform sampler2D photo, illus, depthTex;
uniform bool hasIllus, useDepthTex;
uniform float depthN, opacity, highlight;
uniform float styleFrom, styleTo, sweep, sepia, sat, dawn, time;
uniform vec3 tint;
varying vec2 vUv;
void main() {
  vec4 p = texture2D(photo, vUv);
  vec4 i = hasIllus ? texture2D(illus, vUv) : p;
  // paint-over: the style change sweeps from far (sky, horizon) to near
  float dn = useDepthTex ? texture2D(depthTex, vUv).r : depthN;
  float s = mix(styleFrom, styleTo, smoothstep(0.0, 0.2, sweep * 1.25 - (1.0 - dn)));
  vec4 c = mix(p, i, s);
  float l = dot(c.rgb, vec3(0.299, 0.587, 0.114));
  vec3 col = vec3(l) + (c.rgb - vec3(l)) * sat;
  col = mix(col, l * vec3(1.07, 0.95, 0.78), sepia) * tint;
  col = mix(col * vec3(0.28, 0.33, 0.5), col, dawn);            // pre-dawn -> dawn grade
  col = mix(col, vec3(1.0, 0.85, 0.4), highlight * (0.35 + 0.15 * sin(time * 6.0)));
  gl_FragColor = vec4(col, c.a * opacity);
}`;

const loader = new THREE.TextureLoader();
let pending = 0, loaded = 0;
function tex(path) {
  if (!path) return null;
  pending++;
  const t = loader.load(new URL(path, sceneUrl).href, () => { loaded++; status(); }, undefined,
    () => { loaded++; console.warn('missing', path); status(); });
  t.colorSpace = THREE.NoColorSpace;
  t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  return t;
}

function material(entry, depthN, depthTex) {
  const photo = tex(entry.photo);
  const illus = tex(entry.illus);
  return new THREE.ShaderMaterial({
    vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthTest: false, depthWrite: false,
    side: THREE.DoubleSide,
    uniforms: {
      ...U, photo: { value: photo }, illus: { value: illus || photo }, hasIllus: { value: !!illus },
      depthTex: { value: depthTex || photo }, useDepthTex: { value: !!depthTex },
      depthN: { value: depthN }, opacity: { value: 1 }, highlight: { value: 0 },
    },
  });
}

// ---------------------------------------------------------------- scene building
let scene, W, H;
const layers = [];   // { def, mat, meshes }
const birds = [];    // { def, mat, sprites, next, singingUntil }

function addLayer(def, mat, bbox, order) {
  const [x0, y0, w, h] = bbox;
  const g = new THREE.SphereGeometry(R, Math.max(8, Math.ceil(256 * w / W)), Math.max(4, Math.ceil(128 * h / H)),
    2 * Math.PI * x0 / W, 2 * Math.PI * w / W, Math.PI * y0 / H, Math.PI * h / H);
  g.scale(-1, 1, 1);
  const sphere = new THREE.Mesh(g, mat);
  sphere.renderOrder = order;
  lookScene.add(sphere);
  // flat equirect view: the panorama is the rectangle x -1..1, y -0.5..0.5
  const meshes = [sphere];
  const pg = new THREE.PlaneGeometry(2 * w / W, h / H);
  const cx = -1 + 2 * (x0 + w / 2) / W, cy = 0.5 - (y0 + h / 2) / H;
  for (const shift of [0, -2, 2]) {   // copies either side so layers crossing the seam wrap
    if (shift && !(x0 + w > W && shift === -2) && !(x0 < 0 && shift === 2)) continue;
    const m = new THREE.Mesh(pg, mat);
    m.position.set(cx + shift, cy, 0);
    m.renderOrder = order;
    flatScene.add(m);
    meshes.push(m);
  }
  return meshes;
}

const depthN = d => Math.min(1, Math.log(Math.max(d, 1)) / Math.log(10000));

async function load() {
  const res = await fetch(sceneUrl);
  if (!res.ok) throw new Error(`could not load ${sceneUrl} (${res.status})`);
  scene = await res.json();
  [W, H] = scene.size;
  document.getElementById('title').textContent = scene.title;
  document.title = scene.title;

  const full = [0, 0, W, H];
  const skyMat = material(scene.sky, 1, null);
  addLayer(scene.sky, skyMat, full, 0);
  const plateMat = material(scene.plate, 0.5, tex(scene.plate.depth));
  addLayer(scene.plate, plateMat, full, 1);
  scene.layers.forEach((def, i) => {
    const mat = material(def, depthN(def.depth_m), null);
    layers.push({ def, mat, meshes: addLayer(def, mat, def.bbox, 2 + i), hidden: false, anim: null });
  });
  buildBirds();
  buildEraButtons();
  buildLayerPanel();
  const era = Number(params.get('era')) || scene.capture_year || scene.eras[0];
  setEra(scene.eras.includes(era) ? era : scene.eras[0], true);
  setStyle(Number(params.get('style') || 0));
  document.getElementById('dawn').value = U.dawn.value;
  setMode(mode);
  if (params.get('ui') === '0') document.getElementById('ui').classList.add('hidden');
  if (!scene.plate.filled) console.warn('plate not filled yet: removed layers will still show in the plate');
}

function status() {
  const sub = document.getElementById('subtitle');
  const where = scene?.location?.lat != null ? ` · ${scene.location.lat.toFixed(4)}, ${scene.location.lon.toFixed(4)}` : '';
  sub.textContent = loaded < pending ? `Loading layers ${loaded}/${pending}…`
    : `${eraLabel(currentEra)}${where} · ${layers.length} layers · ${birds.filter(visibleBird).length} birds`;
  if (loaded >= pending) document.body.dataset.ready = '1';
}

// ---------------------------------------------------------------- eras
let currentEra = null;
let lookAnim = null;
const present = (def, year) => {
  const [a, b] = def.lifespan || [null, null];
  return (a == null || year >= a) && (b == null || year <= b);
};
const eraLabel = y => scene?.era_looks?.[y]?.label || String(y);

function setEra(year, instant = false) {
  const now = performance.now();
  const going = [], coming = [];
  for (const L of layers) {
    const target = (!L.hidden && present(L.def, year)) ? 1 : 0;
    const cur = L.mat.uniforms.opacity.value;
    if (Math.abs(target - cur) < 0.01 && !L.anim) continue;
    (target ? coming : going).push([L, target]);
  }
  // things disappear nearest first, then things appear farthest first
  going.sort((a, b) => a[0].def.depth_m - b[0].def.depth_m);
  coming.sort((a, b) => b[0].def.depth_m - a[0].def.depth_m);
  const step = Math.min(140, 1600 / Math.max(1, going.length + coming.length));
  [...going, ...coming].forEach(([L, target], k) => {
    if (instant) { L.mat.uniforms.opacity.value = target; L.anim = null; return; }
    L.anim = { from: L.mat.uniforms.opacity.value, to: target, start: now + k * step, dur: 900 };
  });
  const look = scene.era_looks?.[year] || {};
  const to = { sepia: look.sepia ?? 0, sat: look.sat ?? 1, tint: new THREE.Vector3(...(look.tint || [1, 1, 1])) };
  if (instant) { U.sepia.value = to.sepia; U.sat.value = to.sat; U.tint.value.copy(to.tint); }
  else lookAnim = { from: { sepia: U.sepia.value, sat: U.sat.value, tint: U.tint.value.clone() }, to, start: now, dur: 2500 };
  currentEra = year;
  for (const b of birds) b.target = visibleBird(b) ? 1 : 0;
  document.querySelectorAll('#eras button').forEach(btn => btn.classList.toggle('on', Number(btn.dataset.era) === year));
  refreshLayerStates();
  closeCard();
  status();
}

function buildEraButtons() {
  const box = document.getElementById('eras');
  box.innerHTML = '';
  [...scene.eras].sort((a, b) => a - b).forEach(y => {
    const b = document.createElement('button');
    b.textContent = eraLabel(y);
    b.dataset.era = y;
    b.onclick = () => setEra(y);
    box.appendChild(b);
  });
}

// ---------------------------------------------------------------- style + paint over
let paintAnim = null;
function setStyle(v) {
  U.styleFrom.value = U.styleTo.value = v;
  U.sweep.value = 1;
  document.getElementById('style').value = v;
}
function paintOver() {
  const from = U.styleTo.value, to = from < 0.5 ? 1 : 0;
  U.styleFrom.value = from; U.styleTo.value = to; U.sweep.value = 0;
  paintAnim = { start: performance.now(), dur: 6000, to };
}

// ---------------------------------------------------------------- birds
function glowTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d');
  const grd = g.createRadialGradient(64, 64, 4, 64, 64, 62);
  grd.addColorStop(0, 'rgba(255,248,220,1)');
  grd.addColorStop(0.25, 'rgba(255,214,120,0.9)');
  grd.addColorStop(1, 'rgba(255,190,80,0)');
  g.fillStyle = grd; g.fillRect(0, 0, 128, 128);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

const visibleBird = b => b.def.era === currentEra &&
  (!b.def.layer || layers.find(L => L.def.id === b.def.layer && !L.hidden && present(L.def, currentEra)));

function buildBirds() {
  const glow = glowTexture();
  for (const def of scene.birds || []) {
    const mat = new THREE.SpriteMaterial({ map: glow, transparent: true, depthTest: false, opacity: 0 });
    const sizeDeg = Math.min(2.4, Math.max(0.7, 7 / Math.sqrt(Math.max(def.depth_m, 1))));
    const look = new THREE.Sprite(mat);
    look.position.copy(dir(def.yaw, def.pitch).multiplyScalar(R * 0.9));
    look.scale.setScalar(2 * R * 0.9 * Math.tan(sizeDeg * DEG / 2));
    look.renderOrder = 10000;
    const flat = new THREE.Sprite(mat);
    flat.position.set(def.yaw / 180, def.pitch / 180, 1);
    flat.scale.setScalar(sizeDeg / 180);
    flat.renderOrder = 10000;
    lookScene.add(look); flatScene.add(flat);
    const b = { def, mat, sprites: [look, flat], base: [look.scale.x, flat.scale.x], target: 0,
      next: performance.now() + 1000 + Math.random() * 8000, singingUntil: 0 };
    look.userData.bird = flat.userData.bird = b;
    birds.push(b);
  }
}

// ---------------------------------------------------------------- audio
// Synth chirps stand in for real recordings until the soundscape's own stems are wired in
// (bird.audio in scene.json). Each bird is an HRTF-panned source in the panorama's 3D frame.
let actx = null, master = null, soloBird = null;
const buffers = new Map();

function startAudio() {
  if (actx) { actx.state === 'running' ? actx.suspend() : actx.resume(); updateSoundButton(); return; }
  actx = new AudioContext();
  master = actx.createGain();
  master.gain.value = 0.8;
  master.connect(actx.destination);
  updateSoundButton();
}
function updateSoundButton() {
  const on = actx && actx.state === 'running';
  const b = document.getElementById('btnSound');
  b.textContent = on ? 'Sound on' : 'Sound off';
  b.classList.toggle('on', !!on);
}

function hash(s) { let h = 2166136261; for (const c of s) h = Math.imul(h ^ c.charCodeAt(0), 16777619); return (h >>> 0) / 4294967296; }

async function sing(b) {
  if (!actx || actx.state !== 'running') return;
  const panner = new THREE.Vector3().copy(dir(b.def.yaw, b.def.pitch)).multiplyScalar(Math.min(Math.max(b.def.depth_m, 2), 200));
  const p = actx.createPanner();
  Object.assign(p, { panningModel: 'HRTF', distanceModel: 'inverse', refDistance: 6, rolloffFactor: 1 });
  p.positionX.value = panner.x; p.positionY.value = panner.y; p.positionZ.value = panner.z;
  const g = actx.createGain();
  g.gain.value = soloBird && soloBird !== b ? 0.12 : 1;
  g.connect(p).connect(master);
  const t0 = actx.currentTime + 0.02;
  if (b.def.audio) {
    const url = new URL(b.def.audio, sceneUrl).href;
    if (!buffers.has(url)) buffers.set(url, fetch(url).then(r => r.arrayBuffer()).then(a => actx.decodeAudioData(a)).catch(() => null));
    const buf = await buffers.get(url);
    if (buf) { const s = actx.createBufferSource(); s.buffer = buf; s.connect(g); s.start(t0); return; }
  }
  const h = hash(b.def.species), base = 1800 + h * 4200, notes = 2 + Math.floor(h * 97 % 5);
  for (let n = 0; n < notes; n++) {
    const o = actx.createOscillator(), e = actx.createGain();
    const t = t0 + n * (0.09 + (h * 13 % 1) * 0.12), d = 0.06 + (h * 7 % 1) * 0.1;
    o.type = 'sine';
    o.frequency.setValueAtTime(base * (1 + 0.25 * Math.sin(n * 2.1 + h * 10)), t);
    o.frequency.exponentialRampToValueAtTime(base * (0.7 + (h * 31 % 1) * 0.8), t + d);
    e.gain.setValueAtTime(0, t);
    e.gain.linearRampToValueAtTime(0.25, t + 0.01);
    e.gain.exponentialRampToValueAtTime(0.001, t + d);
    o.connect(e).connect(g);
    o.start(t); o.stop(t + d + 0.02);
  }
}

// ---------------------------------------------------------------- pull-out card
function openCard(b) {
  soloBird = b;
  const d = b.def;
  document.getElementById('cardName').textContent = d.name;
  const latin = d.species.replace(/_/g, ' ');
  document.getElementById('cardSpecies').textContent = latin.charAt(0).toUpperCase() + latin.slice(1);
  const facts = { Era: eraLabel(d.era), Perch: d.perch, 'On layer': d.layer || 'background', Distance: `~${Math.round(d.depth_m)} m` };
  document.getElementById('cardFacts').innerHTML = Object.entries(facts).map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
  document.getElementById('card').hidden = false;
  if (mode === 'look') turnTo(d.yaw, d.pitch);
  b.next = performance.now();
}
function closeCard() { soloBird = null; document.getElementById('card').hidden = true; }

let turnAnim = null;
function turnTo(yaw, pitch) {
  let dy = ((yaw - view.yaw + 540) % 360) - 180;
  turnAnim = { from: { ...view }, dy, dp: pitch - view.pitch, start: performance.now(), dur: 1200 };
}

// ---------------------------------------------------------------- layer panel
function buildLayerPanel() {
  const list = document.getElementById('layerList');
  list.innerHTML = '';
  for (const L of [...layers].reverse()) {   // nearest at the top
    const d = L.def, row = document.createElement('div');
    row.className = 'layer';
    const research = d.lifespan_source === 'default' ? '<span class="flag" title="Lifespan is a class default - needs research">●</span>' : '';
    row.innerHTML = `
      <input type="checkbox" checked title="Show/hide regardless of era">
      <span class="name">${research} ${d.id}</span><span class="state"></span>
      <div class="meta">${Math.round(d.depth_m)} m · from <input type="number" class="from" placeholder="—">
        to <input type="number" class="to" placeholder="—"></div>`;
    const [from, to] = d.lifespan || [null, null];
    row.querySelector('.from').value = from ?? '';
    row.querySelector('.to').value = to ?? '';
    row.querySelector('input[type=checkbox]').onchange = e => { L.hidden = !e.target.checked; setEra(currentEra); };
    for (const cls of ['from', 'to']) {
      row.querySelector('.' + cls).onchange = () => {
        const v = s => (s === '' ? null : Number(s));
        d.lifespan = [v(row.querySelector('.from').value), v(row.querySelector('.to').value)];
        d.lifespan_source = 'edited';
        row.querySelector('.flag')?.remove();
        setEra(currentEra);
      };
    }
    row.onmouseenter = () => { L.mat.uniforms.highlight.value = 1; };
    row.onmouseleave = () => { L.mat.uniforms.highlight.value = 0; };
    L.row = row;
    list.appendChild(row);
  }
}
function refreshLayerStates() {
  for (const L of layers) {
    if (!L.row) continue;
    const s = L.row.querySelector('.state'), on = present(L.def, currentEra);
    s.textContent = on ? 'present' : 'absent';
    s.classList.toggle('present', on);
  }
}
function exportScene() {
  const blob = new Blob([JSON.stringify(scene, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'scene.json';
  a.click();
  URL.revokeObjectURL(a.href);
}

// ---------------------------------------------------------------- modes + input
function setMode(m) {
  mode = m;
  document.getElementById('btnMode').textContent = { look: 'Look', flat360: 'Flat 360°', flat180: 'Flat 180°' }[m];
  resize();
}
function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false);
  lookCam.aspect = w / h;
  lookCam.updateProjectionMatrix();
  fitFlat();
}
function fitFlat() {
  // flat views fit the panorama (or its front half, centred on the current yaw) into the window
  const w = innerWidth, h = innerHeight;
  const fw = mode === 'flat180' ? 1 : 2, fh = 1;
  const s = Math.max(fw / w, fh / h);
  const cx = mode === 'flat180' ? (view.yaw / 180) : 0;
  Object.assign(flatCam, { left: cx - w * s / 2, right: cx + w * s / 2, top: h * s / 2, bottom: -h * s / 2 });
  flatCam.updateProjectionMatrix();
}
addEventListener('resize', resize);

let drag = null;
canvas.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, yaw: view.yaw, pitch: view.pitch, moved: 0 }; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener('pointermove', e => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.moved = Math.max(drag.moved, Math.hypot(dx, dy));
  if (mode !== 'look') return;
  const k = lookCam.fov / innerHeight;
  view.yaw = drag.yaw - dx * k;
  view.pitch = Math.max(-85, Math.min(85, drag.pitch + dy * k));
  turnAnim = null;
});
canvas.addEventListener('pointerup', e => {
  const click = drag && drag.moved < 5;
  drag = null;
  if (!click) return;
  const ray = new THREE.Raycaster();
  ray.setFromCamera(new THREE.Vector2(e.clientX / innerWidth * 2 - 1, -e.clientY / innerHeight * 2 + 1), mode === 'look' ? lookCam : flatCam);
  const sprites = birds.filter(b => b.target > 0).map(b => b.sprites[mode === 'look' ? 0 : 1]);
  const hit = ray.intersectObjects(sprites)[0];
  if (hit) openCard(hit.object.userData.bird); else closeCard();
});
canvas.addEventListener('wheel', e => {
  lookCam.fov = Math.max(25, Math.min(100, lookCam.fov + e.deltaY * 0.05));
  lookCam.updateProjectionMatrix();
}, { passive: true });

document.getElementById('style').oninput = e => setStyle(Number(e.target.value));
document.getElementById('dawn').oninput = e => { U.dawn.value = Number(e.target.value); };
document.getElementById('btnPaint').onclick = paintOver;
document.getElementById('btnMode').onclick = () => setMode(MODES[(MODES.indexOf(mode) + 1) % MODES.length]);
document.getElementById('btnLayers').onclick = () => {
  const p = document.getElementById('layers');
  p.hidden = !p.hidden;
  document.getElementById('btnLayers').classList.toggle('on', !p.hidden);
};
document.getElementById('btnExport').onclick = exportScene;
document.getElementById('btnSound').onclick = startAudio;
document.getElementById('cardClose').onclick = closeCard;
addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  const eras = [...(scene?.eras || [])].sort((a, b) => a - b);
  if (/^[1-9]$/.test(e.key) && eras[e.key - 1]) setEra(eras[e.key - 1]);
  else if (e.key === 'p') paintOver();
  else if (e.key === 'f') document.getElementById('btnMode').click();
  else if (e.key === 'l') document.getElementById('btnLayers').click();
  else if (e.key === 'h') document.getElementById('ui').classList.toggle('hidden');
  else if (e.key === 'Escape') closeCard();
});

// ---------------------------------------------------------------- frame loop
const ease = t => t < 0 ? 0 : t > 1 ? 1 : t * t * (3 - 2 * t);
function frame(now) {
  U.time.value = now / 1000;
  for (const L of layers) {
    if (!L.anim) continue;
    const t = ease((now - L.anim.start) / L.anim.dur);
    L.mat.uniforms.opacity.value = L.anim.from + (L.anim.to - L.anim.from) * t;
    if (t >= 1) L.anim = null;
  }
  if (lookAnim) {
    const t = ease((now - lookAnim.start) / lookAnim.dur), a = lookAnim.from, b = lookAnim.to;
    U.sepia.value = a.sepia + (b.sepia - a.sepia) * t;
    U.sat.value = a.sat + (b.sat - a.sat) * t;
    U.tint.value.lerpVectors(a.tint, b.tint, t);
    if (t >= 1) lookAnim = null;
  }
  if (paintAnim) {
    const t = Math.min(1, (now - paintAnim.start) / paintAnim.dur);
    U.sweep.value = t;
    if (t >= 1) { setStyle(paintAnim.to); paintAnim = null; }
  }
  if (turnAnim) {
    const t = ease((now - turnAnim.start) / turnAnim.dur);
    view.yaw = turnAnim.from.yaw + turnAnim.dy * t;
    view.pitch = turnAnim.from.pitch + turnAnim.dp * t;
    if (t >= 1) turnAnim = null;
  }
  for (const b of birds) {
    b.mat.opacity += (b.target * (soloBird && soloBird !== b ? 0.35 : 1) - b.mat.opacity) * 0.06;
    if (b.target && now > b.next) {
      b.singingUntil = now + 1600;
      b.next = now + 5000 + Math.random() * 9000;
      sing(b);
    }
    const pulse = now < b.singingUntil ? 1 + 0.5 * Math.sin((b.singingUntil - now) / 1600 * Math.PI) : 1;
    const solo = soloBird === b ? 1.8 : 1;
    b.sprites.forEach((s, i) => s.scale.setScalar(b.base[i] * pulse * solo));
  }
  if (actx) {
    const f = dir(view.yaw, view.pitch), L = actx.listener;
    if (L.forwardX) { L.forwardX.value = f.x; L.forwardY.value = f.y; L.forwardZ.value = f.z; L.upX.value = 0; L.upY.value = 1; L.upZ.value = 0; }
    else L.setOrientation(f.x, f.y, f.z, 0, 1, 0);
  }
  if (mode === 'look') {
    lookCam.lookAt(dir(view.yaw, view.pitch));
    renderer.render(lookScene, lookCam);
  } else {
    if (mode === 'flat180') fitFlat();
    renderer.render(flatScene, flatCam);
  }
  requestAnimationFrame(frame);
}

load().then(() => requestAnimationFrame(frame)).catch(err => {
  document.getElementById('subtitle').textContent = err.message;
  console.error(err);
});
