// 战术讲解标注层：箭头 / 区域突显 / 球员连线 / 聚焦光柱 / 文字标签 / 球员号码牌。
// 全部由 play.annotations 数据驱动，按时间窗淡入淡出、箭头逐段画出。
//
// 锚点 anchor：球员 id（跟随移动）或 [x, z]（固定点）。
//   { type:'arrow', t:[t0,t1], path:{ who, t:[a,b] } | from:anchor, to:anchor, bend, dashed, color, label }
//   { type:'zone',  t, shape:'circle', at:anchor, r | shape:'poly', points:[[x,z]...] | preset:'paint', color, label }
//   { type:'link',  t, ids:[...], closed, color, label }
//   { type:'focus', t, who, color, label }
//   { type:'label', t, at:anchor, text, color, height }
// 号码牌：players[].num（球衣号）+ 可选 players[].tag（姓名条，见 roster.js）。
import * as THREE from 'three';
import { sampleMotion } from './motion.js';

const PALETTE = { off: 0x35e6ff, def: 0xff9a2e, warn: 0xff3b5c, good: 0x5dff8a, gold: 0xffd34d, ball: 0xf2fbff };
const FADE = 0.35;
const DRAW = 0.6;
const PRESETS = {
  paint: [[-2.44, 0], [2.44, 0], [2.44, 5.79], [-2.44, 5.79]],
};

const colorOf = (c) => new THREE.Color(PALETTE[c] ?? c ?? PALETTE.ball);
const clamp01 = (x) => Math.min(1, Math.max(0, x));

const STRIP_VS = /* glsl */ `
  attribute float s;
  attribute float solid;
  varying float vS, vSolid;
  void main() { vS = s; vSolid = solid; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const STRIP_FS = /* glsl */ `
  uniform vec3 uColor;
  uniform float uOpacity, uDash, uTime;
  varying float vS, vSolid;
  void main() {
    if (uDash > 0.5 && vSolid < 0.5 && fract((vS - uTime * 0.8) / 0.42) > 0.55) discard;
    gl_FragColor = vec4(uColor * 1.25, uOpacity);
  }`;
const ZONE_VS = /* glsl */ `
  varying vec3 vW;
  void main() { vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`;
const ZONE_FS = /* glsl */ `
  uniform vec3 uColor;
  uniform float uOpacity, uTime;
  varying vec3 vW;
  void main() {
    float stripe = step(0.5, fract((vW.x + vW.z) * 2.2 - uTime * 0.6));
    gl_FragColor = vec4(uColor, (0.045 + 0.025 * stripe) * uOpacity);
  }`;
const BEAM_FS = /* glsl */ `
  uniform vec3 uColor;
  uniform float uOpacity;
  varying vec2 vUv;
  void main() { gl_FragColor = vec4(uColor, pow(1.0 - vUv.y, 2.0) * 0.35 * uOpacity); }`;
const BEAM_VS = /* glsl */ `
  varying vec2 vUv;
  void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;

/** 地面条带（可带箭头）。pts: Vector2(x,z) 列表。 */
function stripGeometry(pts, width, y, head) {
  const pos = [];
  const sArr = [];
  const solid = [];
  const idx = [];
  let s = 0;
  for (let i = 0; i < pts.length; i++) {
    if (i > 0) s += pts[i].distanceTo(pts[i - 1]);
    const a = pts[Math.max(0, i - 1)];
    const b = pts[Math.min(pts.length - 1, i + 1)];
    const d = new THREE.Vector2().subVectors(b, a).normalize();
    const n = new THREE.Vector2(-d.y, d.x).multiplyScalar(width / 2);
    pos.push(pts[i].x + n.x, y, pts[i].y + n.y, pts[i].x - n.x, y, pts[i].y - n.y);
    sArr.push(s, s);
    solid.push(0, 0);
    if (i > 0) {
      const k = (i - 1) * 2;
      idx.push(k, k + 1, k + 2, k + 1, k + 3, k + 2);
    }
  }
  if (head) {
    const tip = head.tip;
    const d = head.dir;
    const n = new THREE.Vector2(-d.y, d.x);
    const base = tip.clone().addScaledVector(d, -head.len);
    const k = pos.length / 3;
    for (const p of [tip, base.clone().addScaledVector(n, head.w / 2), base.clone().addScaledVector(n, -head.w / 2)]) {
      pos.push(p.x, y + 0.001, p.y);
      sArr.push(s);
      solid.push(1);
    }
    idx.push(k, k + 1, k + 2);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('s', new THREE.Float32BufferAttribute(sArr, 1));
  g.setAttribute('solid', new THREE.Float32BufferAttribute(solid, 1));
  g.setIndex(idx);
  return g;
}

/** 截取折线的前 p 比例（按弧长）。 */
function cutPolyline(pts, p) {
  if (p >= 1) return pts;
  const lens = [0];
  for (let i = 1; i < pts.length; i++) lens.push(lens[i - 1] + pts[i].distanceTo(pts[i - 1]));
  const target = lens[lens.length - 1] * p;
  const out = [pts[0]];
  for (let i = 1; i < pts.length; i++) {
    if (lens[i] <= target) { out.push(pts[i]); continue; }
    const f = (target - lens[i - 1]) / (lens[i] - lens[i - 1] || 1);
    out.push(pts[i - 1].clone().lerp(pts[i], f));
    break;
  }
  return out;
}

const labelCache = new Map();
function labelTexture(text, color) {
  const key = `${text}|${color.getHexString()}`;
  if (labelCache.has(key)) return labelCache.get(key);
  const cv = document.createElement('canvas');
  const ctx = cv.getContext('2d');
  const font = '600 44px "PingFang SC","Noto Sans CJK SC","Microsoft YaHei",sans-serif';
  ctx.font = font;
  const w = Math.ceil(ctx.measureText(text).width) + 48;
  cv.width = w;
  cv.height = 72;
  ctx.font = font;
  ctx.fillStyle = 'rgba(6,14,20,0.82)';
  ctx.strokeStyle = `#${color.getHexString()}`;
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.roundRect(2, 2, w - 4, 68, 14);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = '#f2fbff';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 24, 38);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  const out = { tex, aspect: w / 72 };
  labelCache.set(key, out);
  return out;
}

/** 球员号码牌：圆形号码，有 player 姓名时在右侧接一段姓名条。 */
function badgeTexture(num, player, color) {
  const hex = `#${color.getHexString()}`;
  const cv = document.createElement('canvas');
  const ctx = cv.getContext('2d');
  const font = '600 44px "PingFang SC","Noto Sans CJK SC","Microsoft YaHei",system-ui,sans-serif';
  ctx.font = font;
  const nameW = player ? Math.ceil(ctx.measureText(player).width) + 32 : 0;
  const numW = num ? 94 : 14;   // no jersey known: name/team pill only
  cv.width = numW + nameW;
  cv.height = 88;
  ctx.fillStyle = 'rgba(6,14,20,0.94)';
  ctx.strokeStyle = hex;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(1, 5, cv.width - 2, 76, 13);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = hex;
  ctx.fillRect(2,22,3,40);
  ctx.fillStyle = 'rgba(111,151,168,.22)';
  if (num) { ctx.beginPath(); ctx.roundRect(13,16,65,54,10); ctx.fill(); }
  ctx.fillStyle = '#ffffff';
  ctx.textBaseline = 'middle';
  ctx.font = `700 ${num.length > 1 ? 33 : 36}px system-ui,sans-serif`;
  ctx.textAlign = 'center';
  ctx.fillText(num, 45, 45);
  ctx.font = font; ctx.textAlign = 'left';
  if (player) ctx.fillText(player, numW - 3, 45);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  return { tex, aspect: cv.width / cv.height };
}

function makeLabel(text, color, height = 0.42) {
  const { tex, aspect } = labelTexture(text, color);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false, opacity: 0 }));
  sp.scale.set(height * aspect, height, 1);
  sp.renderOrder = 20;
  return sp;
}

export class TacticsLayer {
  constructor(play, timeline, solver, shared, teamColors, resolveBody = null) {
    this.tl = timeline;
    this.motion = play.motion;
    this.resolveBody = resolveBody;
    this.solver = solver;
    this.shared = shared;
    this.group = new THREE.Group();
    this.items = (play.annotations ?? []).map((a) => this._build(a, teamColors));
    this.badges = play.players.map((p) => {
      const { tex, aspect } = badgeTexture(p.num ?? p.id, p.tag, teamColors[p.team]);
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false }));
      sp.scale.set(0.42 * aspect, 0.42, 1);   // film size
      sp.center.set(0.5, 0.5);
      sp.renderOrder = 19;
      this.group.add(sp);
      return { id: p.id, sp };
    });
  }

  _stripMaterial(color, dashed) {
    return new THREE.ShaderMaterial({
      vertexShader: STRIP_VS, fragmentShader: STRIP_FS,
      uniforms: { uColor: { value: color }, uOpacity: { value: 0 }, uDash: { value: dashed ? 1 : 0 }, uTime: this.shared.uTime },
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    });
  }

  _build(a, teamColors) {
    const color = teamColors[a.color] ?? colorOf(a.color);
    const item = { a, color, objs: [] };
    const add = (o) => { this.group.add(o); item.objs.push(o); return o; };
    if (a.type === 'arrow' || a.type === 'link') {
      item.mesh = add(new THREE.Mesh(new THREE.BufferGeometry(), this._stripMaterial(color, a.dashed ?? a.type === 'link')));
      item.mesh.renderOrder = 5;
    }
    if (a.type === 'zone') {
      const mat = new THREE.ShaderMaterial({
        vertexShader: ZONE_VS, fragmentShader: ZONE_FS,
        uniforms: { uColor: { value: color }, uOpacity: { value: 0 }, uTime: this.shared.uTime },
        transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
      });
      const pts = a.preset ? PRESETS[a.preset] : a.points;
      const geo = a.shape === 'circle' || (!pts && a.r)
        ? new THREE.CircleGeometry(1, 64)
        : new THREE.ShapeGeometry(new THREE.Shape(pts.map(([x, z]) => new THREE.Vector2(x, -z))));
      item.fill = add(new THREE.Mesh(geo, mat));
      item.fill.rotation.x = -Math.PI / 2;
      item.fill.position.y = 0.015;
      item.outlinePts = (pts ? [...pts, pts[0]] : null)?.map(([x, z]) => new THREE.Vector2(x, z));
      item.mesh = add(new THREE.Mesh(new THREE.BufferGeometry(), this._stripMaterial(color, true)));
    }
    if (a.type === 'focus') {
      item.rings = [0.62, 0.8].map((r) => {
        const m = add(new THREE.Mesh(new THREE.RingGeometry(r, r + 0.045, 64, 1, 0, Math.PI * 1.6),
          new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide })));
        m.rotation.x = -Math.PI / 2;
        return m;
      });
      item.beam = add(new THREE.Mesh(new THREE.CylinderGeometry(0.62, 0.62, 3.2, 40, 1, true), new THREE.ShaderMaterial({
        vertexShader: BEAM_VS, fragmentShader: BEAM_FS, uniforms: { uColor: { value: color }, uOpacity: { value: 0 } },
        transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
      })));
    }
    const text = a.type === 'label' ? a.text : a.label;
    if (text) item.label = add(makeLabel(text, color, a.height ?? 0.42));
    return item;
  }

  _anchor(anchor, t) {
    if (typeof anchor === 'string') {
      if (anchor === 'ball') {
        const b = this.solver.at(t);
        return new THREE.Vector2(b.x, b.z);
      }
      const [x, z] = this.tl.pos(anchor, t);
      return new THREE.Vector2(x, z);
    }
    return new THREE.Vector2(anchor[0], anchor[1]);
  }

  _arrowPoints(a, t) {
    if (a.path) {
      const [t0, t1] = a.path.t;
      const N = Math.max(16, Math.round((t1 - t0) * 30));
      return Array.from({ length: N + 1 }, (_, i) => this._anchor(a.path.who, t0 + ((t1 - t0) * i) / N));
    }
    // from/to：起点终点各让出球员半径，中间二次贝塞尔弯曲
    let p0 = this._anchor(a.from, t);
    let p2 = this._anchor(a.to, t);
    const d = p2.clone().sub(p0);
    const len = d.length() || 1;
    d.divideScalar(len);
    if (typeof a.from === 'string') p0 = p0.addScaledVector(d, 0.55);
    if (typeof a.to === 'string') p2 = p2.addScaledVector(d, -0.6);
    const ctrl = p0.clone().add(p2).multiplyScalar(0.5).add(new THREE.Vector2(-d.y, d.x).multiplyScalar((a.bend ?? 0) * len));
    return Array.from({ length: 25 }, (_, i) => {
      const s = i / 24;
      return p0.clone().multiplyScalar((1 - s) ** 2).addScaledVector(ctrl, 2 * s * (1 - s)).addScaledVector(p2, s * s);
    });
  }

  _setStrip(item, pts, width, head) {
    item.mesh.geometry.dispose();
    item.mesh.geometry = stripGeometry(pts, width, 0.02, head);
  }

  update(t, visible, showBadges) {
    const shared = this.shared.uTime.value;
    for (const item of this.items) {
      const { a } = item;
      const [t0, t1] = a.t;
      const fade = visible ? clamp01(Math.min((t - t0) / FADE, (t1 - t) / FADE)) : 0;
      item.objs.forEach((o) => { o.visible = fade > 0; });
      if (fade <= 0) continue;
      const draw = clamp01((t - t0) / DRAW);
      let labelAt = null;

      if (a.type === 'arrow') {
        const full = this._arrowPoints(a, t);
        const pts = cutPolyline(full, draw);
        const tip = pts[pts.length - 1];
        const dir = tip.clone().sub(pts[Math.max(0, pts.length - 2)]).normalize();
        const w = a.width ?? 0.09;
        // 箭身在箭头底部截止，避免与箭头叠加发亮
        let total = 0;
        for (let i = 1; i < pts.length; i++) total += pts[i].distanceTo(pts[i - 1]);
        const body = cutPolyline(pts, clamp01((total - w * 3.5) / (total || 1)));
        this._setStrip(item, body, w, { tip, dir, len: w * 4, w: w * 4.2 });
        item.mesh.material.uniforms.uOpacity.value = fade * 0.95;
        labelAt = full[Math.floor(full.length / 2)];
      } else if (a.type === 'link') {
        const pts = a.ids.map((id) => this._anchor(id, t));
        if (a.closed) pts.push(pts[0].clone());
        this._setStrip(item, cutPolyline(pts, draw), a.width ?? 0.05);
        item.mesh.material.uniforms.uOpacity.value = fade * 0.9;
        labelAt = pts.slice(0, a.ids.length).reduce((s, p) => s.add(p), new THREE.Vector2()).divideScalar(a.ids.length);
      } else if (a.type === 'zone') {
        const c = a.at ? this._anchor(a.at, t) : null;
        const grow = 0.6 + 0.4 * (1 - (1 - draw) ** 3);
        if (c) {
          item.fill.position.set(c.x, 0.015, c.y);
          item.fill.scale.setScalar(a.r * grow);
          const ring = Array.from({ length: 65 }, (_, i) => c.clone().add(new THREE.Vector2(Math.cos(i / 64 * Math.PI * 2), Math.sin(i / 64 * Math.PI * 2)).multiplyScalar(a.r * grow)));
          this._setStrip(item, ring, 0.05);
          labelAt = c.clone().add(new THREE.Vector2(0, -a.r * 0.2));
        } else {
          this._setStrip(item, cutPolyline(item.outlinePts, draw), 0.05);
          labelAt = item.outlinePts.slice(0, -1).reduce((s, p) => s.add(p), new THREE.Vector2()).divideScalar(item.outlinePts.length - 1);
        }
        item.fill.material.uniforms.uOpacity.value = fade;
        item.mesh.material.uniforms.uOpacity.value = fade;
      } else if (a.type === 'focus') {
        const c = this._anchor(a.who, t);
        item.rings.forEach((r, i) => {
          r.position.set(c.x, 0.03, c.y);
          r.rotation.z = shared * (i ? -1.2 : 1.6);
          r.scale.setScalar(0.8 + 0.2 * draw);
          r.material.opacity = fade * 0.9;
        });
        item.beam.position.set(c.x, 1.6, c.y);
        item.beam.material.uniforms.uOpacity.value = fade;
        labelAt = c;
      } else if (a.type === 'label') {
        labelAt = this._anchor(a.at, t);
      }

      if (item.label && labelAt) {
        const h = a.type === 'focus' || typeof a.at === 'string' ? 2.75 : (a.y ?? 0.6);
        item.label.position.set(labelAt.x, h, labelAt.y);
        item.label.material.opacity = fade * clamp01((t - t0 - DRAW * 0.5) / 0.3);
      }
    }
    for (const { id, sp } of this.badges) {
      const pose = this.tl.pose(id, t);
      sp.visible = showBadges;
      // The renderer may blend or fully reject an uncertain monocular pose.
      // Follow the head that was actually drawn in this same absolute-time
      // frame; a raw motion head can otherwise leave the badge at waist level.
      const body = this.resolveBody?.(id);
      const renderedHead = body?.joints?.headBase;
      if (renderedHead && [renderedHead.x,renderedHead.y,renderedHead.z].every(Number.isFinite)) {
        sp.position.copy(renderedHead).add(new THREE.Vector3(0, .45*(body.bodyScale ?? 1), 0));
        continue;
      }
      const motion = sampleMotion(this.motion, id, t);
      const headIndex = motion?.jointNames.indexOf('head') ?? -1;
      const head = headIndex >= 0 && motion.coverageWeight > 0 && motion.confidence[headIndex] >= .3
        ? motion.jointsWorld[headIndex] : null;
      if (head) sp.position.set(head[0], head[1] + .3, head[2]);
      else sp.position.set(pose.x, 2.35 + pose.jump, pose.z);
    }
  }
}
