// 入口：场景、Bloom 后处理、相机预设、播放控制、录制。
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { buildCourt } from './court.js';
import { HoloPlayer } from './body.js';
import { Timeline } from './timeline.js';
import { Ball, BallSolver } from './ball.js';
import { TacticsLayer } from './tactics.js';
import { PLAYS } from './plays.js';
import { Narrator } from './narration.js';
import { Minimap } from './minimap.js';
import { resolveNames, paletteLabels } from './roster.js';
import { sampleMotion } from './motion.js';
import { actorWindowEnvelope, actorSafeRect, fitActorCamera, applyActorCamera, actorScreenBounds } from './camera-fit.js';

await document.fonts.load('600 44px "AdkCJK"', '战术复盘进攻防守').catch(() => {});

const exportMode = new URLSearchParams(location.search).get('export') === '1';
if (exportMode) document.body.classList.add('export');
const COLORS = { off: new THREE.Color(0x35e6ff), def: new THREE.Color(0xff9a2e) };
const CAMERAS = {
  broadcast: { pos: [0, 8, 17], target: [0, 0.6, 5.2] },
  top: { pos: [0, 24, 6.6], target: [0, 0, 6.4] },
  baseline: { pos: [5.5, 3.2, -4.5], target: [0, 1.2, 6] },
  sideline: { pos: [-17, 9, 8.2], target: [0.6, 0, 7.2] },   // 与转播画面同向：近端边线、篮筐在左
  follow: null,   // 每帧跟球
};

async function loadPlay() {
  const q = new URLSearchParams(location.search);
  if (q.get('data')) {
    const j = await (await fetch(q.get('data'))).json();
    return resolveNames(j, j.roster);
  }
  throw new Error('A source-bound caller scene is required; no built-in/default replay is available');
}

const play = await loadPlay();
if (play.motion && play.motion.sourceSha256 !== play.metadata?.sourceSha256)
  throw new Error('Motion reconstruction is bound to another source video');
const actorAliases = { points: 'anatomical-points', xbot: 'xbot-points', mesh: 'xbot-mesh' };
const requestedActorStyle = new URLSearchParams(location.search).get('body') ?? play.actorStyle ?? 'anatomical-points';
const actorStyle = actorAliases[requestedActorStyle] ?? requestedActorStyle;
if (!['anatomical-points', 'xbot-points', 'xbot-mesh'].includes(actorStyle))
  throw new Error('Unknown actor style: ' + requestedActorStyle);
let AvatarPlayer = null;
if (actorStyle !== 'anatomical-points') {
  const avatar = await import('./avatar.js');
  await avatar.loadAvatar(new URL('./models/Xbot.glb', location.href).href);
  AvatarPlayer = avatar.AvatarPlayer;
}
const $ = (id) => document.getElementById(id);

// ---------- 渲染器与场景 ----------
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
$('stage').appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x020406);
scene.fog = new THREE.Fog(0x020406, 30, 70);
const camera = new THREE.PerspectiveCamera(40, innerWidth / innerHeight, 0.1, 200);
camera.position.set(...CAMERAS.broadcast.pos);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(...CAMERAS.broadcast.target);
controls.enableDamping = !exportMode;
controls.maxPolarAngle = Math.PI * 0.49;
controls.minDistance = 3;
controls.maxDistance = 60;
controls.autoRotateSpeed = 0.5;

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.36, 0.28, 0.68);
composer.addPass(bloom);
composer.addPass(new OutputPass());

const shared = { uTime: { value: 0 }, uScale: { value: 1 } };
function resize() {
  renderer.setSize(innerWidth, innerHeight);
  composer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  const h = renderer.getDrawingBufferSize(new THREE.Vector2()).y;
  shared.uScale.value = h / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
}
addEventListener('resize', resize);
resize();

scene.add(buildCourt({ fullCourt: play.fullCourt === true }));

// ---------- 战术对象 ----------
const timeline = new Timeline(play);
const players = play.players.map((p) => {
  const heightMeters = play.motion?.players?.[p.id]?.heightMeters ?? p.heightMeters ?? 1.88;
  const hp = new HoloPlayer(COLORS[p.team], shared, heightMeters);
  hp.disc.visible = play.exportMinimalHUD !== true;
  const visual = AvatarPlayer ? new AvatarPlayer(COLORS[p.team], shared, { style: actorStyle, heightMeters }) : null;
  if (visual) {
    if (visual.disc) visual.disc.visible = false;
    scene.add(visual.group, hp.disc);
  } else scene.add(hp.group);
  return { def: p, hp, visual };
});
const solver = new BallSolver(play, timeline, new HoloPlayer(COLORS.off, shared));
for(const {def,hp} of players)hp.prepareAppearanceFrames(
  (time)=>({...timeline.pose(def.id,time),ballContacts:solver.contactsAt(def.id,time)}),
  (time)=>sampleMotion(play.motion,def.id,time),play.duration);
const ball = new Ball();
scene.add(ball.group, ball.trail);
scene.add(ball.shadow);
scene.add(new THREE.HemisphereLight(0xbfe9ff, 0x101820, 1.2));
const key = new THREE.DirectionalLight(0xffffff, 1.6);
key.position.set(4, 10, 12);
scene.add(key);
const tactics = new TacticsLayer(play, timeline, solver, shared, COLORS,
  (id) => { const actor = players.find(({def}) => def.id === id); return actor?.visual ?? actor?.hp; });
scene.add(tactics.group);

$('title').textContent = play.title;
for (let px = 22; $('title').scrollWidth > $('title').clientWidth && px > 14; ) $('title').style.fontSize = `${--px}px`;
const legendLabels = paletteLabels(play);
for (const key of ['off','def'])
  document.querySelector(`[data-palette-label="${key}"]`).textContent = legendLabels[key];

// ---------- 语音解说 / 站位分析 / 原视频画中画 ----------
const narrator = new Narrator(exportMode ? { captions: [] } : play);
narrator.ready.then(() => {
  if (!narrator.available) return;
  $('voice').hidden = false;
  $('start').hidden = false;
});
function setVoice(on) {
  if (on) narrator.enable(); else narrator.disable();
  $('voice').classList.toggle('on', on);
}
$('start').onclick = () => { $('start').hidden = true; setVoice(true); seekTo(0); state.playing = true; };
$('voice').onclick = () => { $('start').hidden = true; setVoice(!narrator.enabled); };

const minimap = play.phases || play.players.length > 4 ? new Minimap(play, timeline, $('side'), { sampleBall: (time) => solver.at(time) }) : null;
if (minimap) {
  $('side').hidden = false;
  $('sideToggle').hidden = false;
  document.querySelector('.hint').hidden = true;
  $('sideOn').onchange = (e) => { $('side').hidden = !e.target.checked; };
  if (play.exportMinimalHUD) {
    $('side').querySelector('[data-k=stats]').hidden = true;
    $('side').querySelector('[data-k=intent]').hidden = true;
  }
}

const pip = (!exportMode || play.exportPip) && play.video ? $('pip').querySelector('video') : null;
if (pip) {
  pip.src = play.video.src;
  if (exportMode) document.body.classList.add('pip');
  $('pip').hidden = false;
  $('pipToggle').hidden = false;
  $('pipOn').onchange = (e) => { $('pip').hidden = !e.target.checked; };
}
/** 画中画跟随剧情时间：播放时按倍率走、漂移 >0.3s 才纠正；暂停/定格时停在对应帧。 */
function syncVideo(moving) {
  if (!pip || $('pip').hidden || pip.readyState < 1) return;
  const target = Math.min(state.t + (play.video.offset ?? 0), pip.duration || Infinity);
  if (moving) {
    pip.playbackRate = Math.min(4, Math.max(0.0625, state.speed * slowRate(state.t)));
    if (pip.paused) pip.play().catch(() => {});
    if (Math.abs(pip.currentTime - target) > 0.3) pip.currentTime = target;
  } else {
    if (!pip.paused) pip.pause();
    if (!pip.seeking && Math.abs(pip.currentTime - target) > 0.05) pip.currentTime = target;
  }
}
const capIndex = (t) => (play.captions ?? []).findIndex(([a, b]) => t >= a && t < b);

// ---------- 播放状态 ----------
const state = { t: 0, playing: !exportMode, speed: 1, notes: true, badges: true, cam: 'broadcast', recorder: null, holdLeft: 0, held: new Set(), waitVoice: false };
const seek = $('seek');
seek.max = play.duration;

let lastRenderedSourceTime = 0;
let actorEnvelope = null, actorCameraFit = null, actorCameraKey = null;
let lastFrameSafety = { safeRect:null, panels:[] };
function renderFrame(t) {
  lastRenderedSourceTime = t;
  for (const { def, hp, visual } of players) {
    const pose = timeline.pose(def.id, t);
    pose.time = t;
    // Apply source-clock palm contact after the tracked pose is retargeted.
    pose.ballContacts = solver.contactsAt(def.id, t);
    const motion = sampleMotion(play.motion, def.id, t);
    if (motion) hp.setTrackedPose(pose, motion, Boolean(visual)); else hp.setPose(pose, Boolean(visual));
    if (visual) visual.updateFromRig(hp);
  }
  ball.update(solver, t);
  tactics.update(t, state.notes, state.badges);
  if (minimap && !$('side').hidden) minimap.update(t, solver.hiddenAt(t) ? null : ball.group.position);

  const cap = (play.captions ?? []).find(([a, b]) => t >= a && t < b);
  const text = cap ? cap[2] : '';
  if ($('caption').textContent !== text) {
    $('caption').textContent = text;
    $('caption').classList.toggle('show', !!text);
  }
  seek.value = t;
  $('time').textContent = `${t.toFixed(1)} / ${play.duration.toFixed(1)}s`;
}

// ---------- 相机 ----------
let camTween = null;
function setCamera(name) {
  if (name === 'follow' && solver.unknown) name = 'broadcast';
  state.cam = name;
  document.querySelectorAll('[data-cam]').forEach((b) => b.classList.toggle('on', b.dataset.cam === name));
  const preset = CAMERAS[name];
  if (!preset) return;
  camTween = {
    t: 0, p0: camera.position.clone(), q0: controls.target.clone(),
    p1: new THREE.Vector3(...preset.pos), q1: new THREE.Vector3(...preset.target),
  };
}

function updateCamera(dt) {
  if (camTween) {
    camTween.t = Math.min(1, camTween.t + dt / 1.2);
    const s = camTween.t * camTween.t * (3 - 2 * camTween.t);
    camera.position.lerpVectors(camTween.p0, camTween.p1, s);
    controls.target.lerpVectors(camTween.q0, camTween.q1, s);
    if (camTween.t >= 1) camTween = null;
  } else if (state.cam === 'follow') {
    const b = ball.group.position;
    controls.target.lerp(new THREE.Vector3(b.x, 1, b.z), 0.06);
    camera.position.lerp(new THREE.Vector3(b.x * 0.6, 5.5, b.z + 9), 0.04);
  }
  controls.update();
}
controls.addEventListener('start', () => {
  camTween = null;
  if (state.cam !== 'free') setCamera('free');
});

// ---------- 录制（一整轮 → webm） ----------
function startRecording() {
  if (state.recorder) return;
  const mime = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'].find((m) => MediaRecorder.isTypeSupported(m));
  const stream = new MediaStream([...renderer.domElement.captureStream(60).getVideoTracks(), ...narrator.audioTracks()]);
  const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 12e6 });
  const chunks = [];
  rec.ondataavailable = (e) => chunks.push(e.data);
  rec.onstop = () => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob(chunks, { type: 'video/webm' }));
    a.download = `holo_${play.title.split(' ')[0]}.webm`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    state.recorder = null;
    $('rec').classList.remove('on');
  };
  seekTo(0);
  state.playing = true;
  rec.start();
  state.recorder = rec;
  $('rec').classList.add('on');
}

/** 跳转：清掉定格状态，让之前的讲解定格可再次触发。 */
function seekTo(t) {
  state.t = t;
  state.holdLeft = 0;
  state.held.clear();
  state.waitVoice = false;
  ball.lastT = null;
  narrator.stop();
  $('caption').classList.remove('hold');
}

// ---------- UI ----------
$('play').onclick = () => { state.playing = !state.playing; };
seek.oninput = () => { seekTo(+seek.value); state.playing = false; };
$('speed').onchange = (e) => { state.speed = +e.target.value; };
$('notes').onchange = (e) => { state.notes = e.target.checked; };
$('badges').onchange = (e) => { state.badges = e.target.checked; };
$('orbit').onchange = (e) => { controls.autoRotate = e.target.checked; };
$('rec').onclick = startRecording;
document.querySelectorAll('[data-cam]').forEach((b) => { b.onclick = () => setCamera(b.dataset.cam); });
addEventListener('keydown', (e) => {
  if (e.code === 'Space') { state.playing = !state.playing; e.preventDefault(); }
  if (e.code === 'ArrowRight') seekTo(Math.min(play.duration, state.t + 0.5));
  if (e.code === 'ArrowLeft') seekTo(Math.max(0, state.t - 0.5));
  const cams = ['broadcast', 'top', 'baseline', 'follow', 'sideline'];
  if (/^Digit[1-5]$/.test(e.code)) setCamera(cams[+e.code.slice(5) - 1]);
});

// 调试/截图用：window.holo.seek(5.2)
window.holo = {
  seek: (t) => { seekTo(t); state.playing = false; renderFrame(t); },
  cam: (pos, target) => { camTween = null; camera.position.set(...pos); controls.target.set(...target); },
  state, setCamera,
};

/** 慢动作倍率：区间两端 0.3s 平滑过渡。 */
function slowRate(t) {
  let r = 1;
  for (const [a, b, k] of play.slow ?? []) {
    const w = Math.min(1, Math.max(0, Math.min((t - a + 0.3) / 0.3, (b + 0.3 - t) / 0.3)));
    r = Math.min(r, 1 + (k - 1) * w);
  }
  return r;
}

/** 推进剧情时间：处理讲解定格（holds）、慢动作（slow）、循环。 */
function advance(dt) {
  if (state.holdLeft > 0) {
    state.holdLeft -= dt * state.speed;
    return;
  }
  const next = state.t + dt * state.speed * slowRate(state.t);
  // 解说没说完就停在这句字幕的末尾等它
  const ci = capIndex(state.t);
  state.waitVoice = narrator.enabled && ci >= 0 && narrator.idx === ci && narrator.busy() && next >= play.captions[ci][1];
  if (state.waitVoice) return;
  const hold = (play.holds ?? []).find(([ht]) => state.t < ht && next >= ht && !state.held.has(ht));
  if (hold) {
    state.t = hold[0];
    state.holdLeft = hold[1];
    state.held.add(hold[0]);
    return;
  }
  state.t = next;
  if (state.t > play.duration) {
    if (state.recorder) state.recorder.stop();
    state.t = 0;
    state.held.clear();
  }
}

let wasPlaying = true;
const clock = new THREE.Clock();
if (!exportMode) renderer.setAnimationLoop(() => {
  const dt = Math.min(clock.getDelta(), 0.05);
  shared.uTime.value += dt;
  if (state.playing) advance(dt);
  if (state.playing !== wasPlaying) {
    wasPlaying = state.playing;
    if (state.playing) narrator.resume(); else narrator.pause();
  }
  if (state.playing && narrator.enabled) {
    const ci = capIndex(state.t);
    if (ci >= 0 && ci !== narrator.idx) narrator.speak(ci, Math.min(2, Math.max(1, state.speed)));
  }
  const holding = state.holdLeft > 0 || state.waitVoice;
  $('caption').classList.toggle('hold', state.playing && holding);
  syncVideo(state.playing && !holding);
  $('play').textContent = state.playing ? '❚❚' : '▶';
  renderFrame(state.t);
  updateCamera(dt);
  composer.render();
});
setCamera(play.camera ?? 'broadcast');

// Movie export is silent. The parent editor owns narration and the final audio clock.
// renderAt(t) is a pure absolute-time frame API, including a repeatable camera orbit.
window.holo.renderAt = async (t, options = {}) => {
  if (!Number.isFinite(t) || t < 0 || t > play.duration) throw new Error('Invalid frame time');
  camTween = null; controls.autoRotate = false;
  state.playing = false; state.notes = options.notes !== false; state.badges = options.badges !== false;
  shared.uTime.value = t;
  const preset = play.cameraPreset ?? CAMERAS[options.camera ?? play.camera ?? 'broadcast'] ?? CAMERAS.broadcast;
  const pos = new THREE.Vector3(...preset.pos), target = new THREE.Vector3(...preset.target);
  const orbit = Number(options.orbitRadians ?? play.exportOrbitRadians ?? 0);
  const frameAllActors = (options.frameAllActors ?? play.exportFrameAllActors) === true;
  const outputProgress = (options.outputTime ?? t) / (options.outputDuration ?? play.exportTimeline?.at(-1)?.[0] ?? play.duration);
  const angle = orbit * (outputProgress - 0.5);
  // Slow push-in over the export: exportPush = fraction of camera distance removed by the end.
  const ease = (u) => u * u * (3 - 2 * u);
  const push = 1 - Number(play.exportPush ?? 0) * ease(Math.min(1, Math.max(0, outputProgress)));
  pos.sub(target).applyAxisAngle(new THREE.Vector3(0,1,0), angle).multiplyScalar(push).add(target);
  if (!frameAllActors) camera.clearViewOffset();
  camera.position.copy(pos); controls.target.copy(target); controls.update();
  $('side').hidden = options.minimap === false;
  renderFrame(t);
  if (play.exportCaptions && play.exportNarrationCaptions !== false) {
    const movieTime = options.outputTime ?? t;
    const cue = play.exportCaptions.find(([a,b]) => movieTime >= a && movieTime < b);
    $('caption').textContent = cue ? cue[2] : '';
    $('caption').classList.toggle('show', !!cue);
  }
  if (play.exportNarrationCaptions === false) {
    $('caption').textContent = ''; $('caption').classList.remove('show');
  }
  if (pip) {
    pip.muted = true; pip.pause();
    if (pip.readyState < 2) await new Promise((resolve,reject) => {
      const timer=setTimeout(()=>reject(new Error('PIP video could not load')),15000);
      pip.addEventListener('loadeddata',()=>{clearTimeout(timer);resolve();},{once:true});
      pip.addEventListener('error',()=>{clearTimeout(timer);reject(new Error('PIP source failed'));},{once:true});
    });
    const target=Math.min(Math.max(0,t+(play.video.offset ?? 0)),Math.max(0,pip.duration-0.001));
    if (Math.abs(pip.currentTime-target)>0.001) await new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>reject(new Error('PIP frame seek timed out')),15000);
      pip.addEventListener('seeked',()=>{clearTimeout(timer);resolve();},{once:true});
      pip.currentTime=target;
    });
  }
  // Scale the complete panels (including text) from their right corners. The
  // 16px right inset stays fixed; cap scale on smaller exports to avoid overlap.
  const requestedScale = Number(options.panelScale ?? play.exportPanelScale ?? 1.25);
  if (!Number.isFinite(requestedScale) || requestedScale < 0.5 || requestedScale > 2)
    throw new Error('exportPanelScale must be between 0.5 and 2');
  const side = $('side'), pipPanel = $('pip');
  const sideVisible = !side.hidden, pipVisible = !!pip && !pipPanel.hidden;
  const available = innerHeight - 16 - (pipVisible ? 115 : 16) - 24;
  const naturalHeight = (sideVisible ? side.offsetHeight : 0) + (pipVisible ? pipPanel.offsetHeight : 0);
  const effectiveScale = Math.min(requestedScale,
    naturalHeight > 0 ? available / naturalHeight : requestedScale,
    (innerWidth - 32) / Math.max(sideVisible ? side.offsetWidth : 0, pipVisible ? pipPanel.offsetWidth : 0, 1));
  document.body.style.setProperty('--export-panel-scale', effectiveScale);
  // Reserve the maximum requested panel width, independent of text-height
  // changes and the current frame's reduced panel scale. Camera fit is cached
  // for the whole observed window and never follows an individual actor.
  const naturalPanelWidth = Math.max(sideVisible ? side.offsetWidth : 0,pipVisible ? pipPanel.offsetWidth : 0);
  const maximumPanelScale = Math.min(requestedScale,(innerWidth-32)/Math.max(1,naturalPanelWidth));
  const reservedPanelWidth = naturalPanelWidth ? 32+naturalPanelWidth*maximumPanelScale : 0;
  const safeRect = frameAllActors ? actorSafeRect(innerWidth,innerHeight,{
    panelWidth:reservedPanelWidth,captions:play.exportNarrationCaptions !== false && !!play.exportCaptions?.length,
  }) : null;
  if (frameAllActors) {
    actorEnvelope ??= actorWindowEnvelope(play);
    const key = JSON.stringify([innerWidth,innerHeight,safeRect,preset.pos,preset.target,camera.fov,orbit]);
    if (key !== actorCameraKey) {
      actorCameraFit = fitActorCamera(actorEnvelope,preset,{width:innerWidth,height:innerHeight,safeRect,fov:camera.fov,orbitRadians:orbit,scale:Number(play.exportFitScale ?? 1)});
      actorCameraKey = key;
    }
    applyActorCamera(camera,controls,actorCameraFit,angle);
  }
  composer.render();
  if (pip) await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
  const bounds = (element, visible) => {
    if (!visible) return null;
    const r = element.getBoundingClientRect();
    return {x:r.x,y:r.y,width:r.width,height:r.height,right:r.right,bottom:r.bottom};
  };
  const minimapBounds=bounds(side,sideVisible),pipBounds=bounds(pipPanel,pipVisible);
  lastFrameSafety={safeRect,panels:[minimapBounds?{...minimapBounds,name:'minimap'}:null,
                                  pipBounds?{...pipBounds,name:'pip'}:null].filter(Boolean)};
  return {time:t,outputTime:options.outputTime ?? t,players:players.length,actorStyle,audio:false,pip:!!pip,pipTime:pip?.currentTime ?? null,
    cameraFraming:frameAllActors?{enabled:true,...actorCameraFit,position:camera.position.toArray(),angle}: {enabled:false},
    panelLayout:{requestedScale,effectiveScale,minimap:minimapBounds,pip:pipBounds,caption:bounds($('caption'),$('caption').classList.contains('show'))}};
};
window.holo.play = play;
window.holo.actorStyle = actorStyle;
window.holo.inspectPresentation = () => ({
  actorCount:players.length,ballObservation:solver.unknown?'unknown':'provided',
  ballGroupVisible:ball.group.visible,ballMeshVisible:ball.mesh.visible,
  ballShadowVisible:ball.shadow.visible,ballTrailVisible:ball.trail.visible,
  ballSolverPosition:solver.at(lastRenderedSourceTime)?.toArray()??null,
  teamRolesVerified:play.metadata?.teamRolesVerified!==false,
  legendTexts:[...document.querySelectorAll('[data-palette-label]')].map(el=>el.textContent),
  minimapLegendTexts:minimap?.legendTexts??[],
  minimapStatsText:minimap?.el.stats.textContent??'',
});
window.holo.inspectActors = () => players.map(({def,hp,visual}) => ({
  id:def.id,actorStyle,bodyScale:(visual??hp).bodyScale,
  screenBounds:actorScreenBounds((visual??hp).group,camera,{width:innerWidth,height:innerHeight,
    ...lastFrameSafety,pixelRatio:renderer.getPixelRatio()}),
  renderGeometry:(()=>{
    const counts={meshes:0,skinnedMeshes:0,pointClouds:0,geometryVertices:0};
    (visual??hp).group.traverse(object=>{
      if(object.isMesh)counts.meshes++;
      if(object.isSkinnedMesh)counts.skinnedMeshes++;
      if(object.isPoints)counts.pointClouds++;
      counts.geometryVertices+=object.geometry?.attributes?.position?.count??0;
    });
    return counts;
  })(),
  controllerHands:['L','R'].map(side=>hp.getPalmPosition(side)?.toArray()),
  displayedHands:['L','R'].map(side=>(visual??hp).getPalmPosition(side)?.toArray()),
  displayedHead:(visual??hp).joints.headBase?.toArray(),
  avatarDiagnostics:visual?.diagnostics??null,
}));
if (exportMode) {
  if (pip) pip.load();
  window.holo.ready = Promise.resolve();
}
