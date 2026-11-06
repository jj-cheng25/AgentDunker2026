// 全息球场：半场线（地面发光条带）、篮架、网格地面、背景星点与大圆弧。
// 坐标系：x = 横向（米，-7.62..7.62），z = 离底线距离（米，0..14.33），y = 高度。
import * as THREE from 'three';

const FT = 0.3048;
export const COURT = {
  halfW: 25 * FT,          // 7.62
  halfLen: 47 * FT,        // 14.33
  hoop: new THREE.Vector3(0, 10 * FT, 5.25 * FT),   // 篮筐中心 (0, 3.05, 1.6)
  rimR: 0.2286,
  boardZ: 4 * FT,
};

const LINE_COLOR = 0xe8f6ff;

/** 把折线变成贴地的细长条带（WebGL 线宽恒为 1px，条带在 bloom 下才有发光感）。 */
function ribbon(points, width, material) {
  const pos = [];
  const idx = [];
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const a = points[Math.max(0, i - 1)];
    const b = points[Math.min(points.length - 1, i + 1)];
    const dx = b.x - a.x;
    const dz = b.z - a.z;
    const len = Math.hypot(dx, dz) || 1;
    const nx = (-dz / len) * width * 0.5;
    const nz = (dx / len) * width * 0.5;
    pos.push(p.x + nx, 0.006, p.z + nz, p.x - nx, 0.006, p.z - nz);
    if (i > 0) {
      const k = (i - 1) * 2;
      idx.push(k, k + 1, k + 2, k + 1, k + 3, k + 2);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  return new THREE.Mesh(g, material);
}

const v = (x, z) => new THREE.Vector3(x, 0, z);

function arc(cx, cz, r, a0, a1, n = 64) {
  const pts = [];
  for (let i = 0; i <= n; i++) {
    const a = a0 + ((a1 - a0) * i) / n;
    pts.push(v(cx + r * Math.cos(a), cz + r * Math.sin(a)));
  }
  return pts;
}

function courtLines(group) {
  const mat = new THREE.MeshBasicMaterial({ color: LINE_COLOR, transparent: true, opacity: 0.55, side: THREE.DoubleSide });
  const W = COURT.halfW;
  const L = COURT.halfLen;
  const hz = COURT.hoop.z;
  const w = 0.05;
  const add = (pts, width = w) => group.add(ribbon(pts, width, mat));

  add([v(-W, L), v(-W, 0), v(W, 0), v(W, L), v(-W, L)]);                  // 边线 + 底线 + 中线
  add([v(-8 * FT, 0), v(-8 * FT, 19 * FT), v(8 * FT, 19 * FT), v(8 * FT, 0)]);   // 禁区
  add([v(-6 * FT, 0), v(-6 * FT, 19 * FT)]);
  add([v(6 * FT, 0), v(6 * FT, 19 * FT)]);
  add(arc(0, 19 * FT, 6 * FT, 0, Math.PI * 2, 72));                     // 罚球圈
  add(arc(0, hz, 4 * FT, 0, Math.PI, 40));                               // 合理冲撞区
  add(arc(0, L, 6 * FT, Math.PI, Math.PI * 2, 48));                      // 中圈（半）

  // 三分线：底角直线 22ft + 弧线 23.75ft
  const r3 = 23.75 * FT;
  const cx = 22 * FT;
  const zc = hz + Math.sqrt(r3 * r3 - cx * cx);
  const a0 = Math.atan2(zc - hz, cx);
  const a1 = Math.PI - a0;
  add([v(cx, 0), v(cx, zc), ...arc(0, hz, r3, a0, a1, 96), v(-cx, zc), v(-cx, 0)]);
}

function hoop(group) {
  const glow = new THREE.LineBasicMaterial({ color: 0xdff4ff, transparent: true, opacity: 0.85 });
  const dim = new THREE.LineBasicMaterial({ color: 0x8fd8c0, transparent: true, opacity: 0.35 });
  const { hoop: h, rimR, boardZ } = COURT;

  // 篮板：外框 + 投篮方框 + 对角线（参考图里的"信封"线）
  const bw = 0.915;
  const by0 = 2.9;
  const by1 = by0 + 1.07;
  const board = [
    [-bw, by0], [bw, by0], [bw, by1], [-bw, by1], [-bw, by0],
  ].map(([x, y]) => new THREE.Vector3(x, y, boardZ));
  group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(board), glow));
  const diag = [[-bw, by1], [0, by0 + 0.45], [bw, by1], [-bw, by0], [0, by0 + 0.45], [bw, by0]];
  group.add(new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(diag.map(([x, y]) => new THREE.Vector3(x, y, boardZ))), dim));
  const sq = [[-0.3, by0 + 0.15], [0.3, by0 + 0.15], [0.3, by0 + 0.6], [-0.3, by0 + 0.6], [-0.3, by0 + 0.15]];
  group.add(new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(sq.map(([x, y]) => new THREE.Vector3(x, y, boardZ + 0.01))), glow));

  // 篮筐（发光环）+ 网（锥形线框）+ 下方投影圆
  const rim = new THREE.Mesh(
    new THREE.TorusGeometry(rimR, 0.012, 8, 48),
    new THREE.MeshBasicMaterial({ color: 0xffffff }));
  rim.rotation.x = Math.PI / 2;
  rim.position.copy(h);
  group.add(rim);

  const net = [];
  const N = 16;
  for (let i = 0; i < N; i++) {
    const a = (i / N) * Math.PI * 2;
    const b = ((i + 2) / N) * Math.PI * 2;
    net.push(
      new THREE.Vector3(h.x + rimR * Math.cos(a), h.y, h.z + rimR * Math.sin(a)),
      new THREE.Vector3(h.x + rimR * 0.55 * Math.cos(b), h.y - 0.42, h.z + rimR * 0.55 * Math.sin(b)));
  }
  group.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(net), dim));
  for (const [y, s] of [[h.y - 0.2, 0.8], [h.y - 0.42, 0.55]]) {
    const ring = new THREE.Line(new THREE.BufferGeometry().setFromPoints(
      arc(0, 0, rimR * s, 0, Math.PI * 2, 32).map((p) => new THREE.Vector3(h.x + p.x, y, h.z + p.z))), dim);
    group.add(ring);
  }
  for (const s of [1.0, 1.5]) {
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(
      arc(h.x, h.z, rimR * s, 0, Math.PI * 2, 40).map((p) => new THREE.Vector3(p.x, 0.01, p.z))), dim));
  }
  // 篮架支柱
  group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, by0 + 0.3, boardZ), new THREE.Vector3(0, by0 + 0.3, -0.9), new THREE.Vector3(0, 0, -0.9),
  ]), dim));
}

function backdrop(group) {
  const grid = new THREE.GridHelper(60, 60, 0x1d4a5a, 0x10303c);
  grid.position.set(0, -0.002, 7);
  grid.material.transparent = true;
  grid.material.opacity = 0.35;
  group.add(grid);

  // 远处的大圆弧（参考图背景里的弧线）
  const ringMat = new THREE.LineBasicMaterial({ color: 0x2a5566, transparent: true, opacity: 0.35 });
  for (const r of [18, 26, 34]) {
    const pts = arc(0, 7, r, 0, Math.PI * 2, 160).map((p) => new THREE.Vector3(p.x, -0.01, p.z));
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), ringMat));
  }

  // 星点
  const n = 500;
  const pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const r = 25 + Math.random() * 40;
    const a = Math.random() * Math.PI * 2;
    pos.set([Math.cos(a) * r, Math.random() * 30 - 2, 7 + Math.sin(a) * r], i * 3);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  group.add(new THREE.Points(g, new THREE.PointsMaterial({ color: 0x6aa6b8, size: 0.12, transparent: true, opacity: 0.6 })));
}

export function buildCourt({ fullCourt = false } = {}) {
  const group = new THREE.Group();
  backdrop(group);
  const half = new THREE.Group();
  courtLines(half);
  hoop(half);
  group.add(half);
  if (fullCourt) {
    // Far half = near half rotated 180° about the centre-court point (0, halfLen).
    const far = half.clone();
    far.rotation.y = Math.PI;
    far.position.z = 2 * COURT.halfLen;
    far.traverse((o) => { if (o.material) { o.material = o.material.clone(); o.material.opacity *= 0.6; } });
    group.add(far);
  }
  return group;
}
