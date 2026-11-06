// 俯视战术板：当前站位距离底纹、固定源时间尾迹与已有战术标注。
// 底纹仅为距离示意；防守三角仅区分队别，不编码预测、控制权或朝向。
import { paletteLabels } from './roster.js';
const W = 15.24;
const L = 14.33;
const HOOP_Z = 1.6;
const OPEN_DIST = 2.2;      // 最近防守人超过这个距离（米）记为空位
const PRESSURE_R = 3.0;     // 持球人周围这个半径内的防守人数 = 受压人数
const CYAN = '#35e6ff';
const ORANGE = '#ff9a2e';
const OPEN_COLOR = '#8bdcff';
const RED = '#ff3b5c';
const GOLD = '#ffd34d';
const TRAIL = 2;
const BALL_TRAIL_HZ = 30;
const FOOT = 76;
const rgba = (hex, a) => `rgba(${parseInt(hex.slice(1, 3), 16)}, ${parseInt(hex.slice(3, 5), 16)}, ${parseInt(hex.slice(5, 7), 16)}, ${a})`;
const who = (p) => p.say ?? `${p.num}号`;

export class Minimap {
  constructor(play, timeline, root, { sampleBall = null } = {}) {
    this.play = play;
    this.rolesVerified = play.metadata?.teamRolesVerified !== false;
    this.paletteLabels = paletteLabels(play);
    this.legendTexts = [];
    this.tl = timeline;
    this.root = root;
    this.sampleBall = sampleBall;
    const sourceTimes = play.exportTimeline?.map((row) => row[1]) ?? [0, play.duration];
    this.sourceStart = Math.max(0, Math.min(...sourceTimes));
    this.sourceEnd = Math.min(play.duration, Math.max(...sourceTimes));
    this.canvas = root.querySelector('canvas');
    // Keep the raster backend stable for export/readback; Chromium otherwise
    // switches from GPU to CPU after early getImageData calls.
    this.g = this.canvas.getContext('2d', { willReadFrequently: true });
    this.el = {
      phase: root.querySelector('[data-k=phase]'),
      intent: root.querySelector('[data-k=intent]'),
      stats: root.querySelector('[data-k=stats]'),
    };
    this.off = play.players.filter((p) => p.team === 'off');
    this.def = play.players.filter((p) => p.team === 'def');
    this.lastPhase = null;
    this.lastStats = '';
    this.heat = document.createElement('canvas');
    this.heat.width = 48;
    this.heat.height = 45;
    this.hg = this.heat.getContext('2d');
    this.heatImg = this.hg.createImageData(48, 45);
  }

  _xy(x, z) {
    const cw = this.canvas.width;
    const ch = this.canvas.height - FOOT;
    const len = this.play.fullCourt ? 2 * L : L;   // full court for transition / backcourt plays
    const s = Math.min(cw / (W + 1.8), ch / (len + 1.2));   // 留出边线外的空间：底角球员的空位光圈不被裁掉
    return [cw / 2 + x * s, 0.6 * s + z * s, s];
  }

  _line(pts, dash = []) {
    const g = this.g;
    g.setLineDash(dash);
    g.beginPath();
    pts.forEach(([x, z], i) => g[i ? 'lineTo' : 'moveTo'](...this._xy(x, z).slice(0, 2)));
    g.stroke();
    g.setLineDash([]);
  }

  _court() {
    const g = this.g;
    const [, , s] = this._xy(0, 0);
    // 油漆区底色 + 三分线内淡色
    g.fillStyle = 'rgba(53, 230, 255, 0.05)';
    const [px0, py0] = this._xy(-2.44, 0);
    g.fillRect(px0, py0, 4.88 * s, 5.79 * s);
    g.strokeStyle = 'rgba(223, 246, 255, 0.4)';
    g.lineWidth = 1;
    this._line([[-W / 2, 0], [W / 2, 0], [W / 2, L], [-W / 2, L], [-W / 2, 0]]);
    this._line([[-2.44, 0], [-2.44, 5.79], [2.44, 5.79], [2.44, 0]]);
    const arc = (cx, cz, r, a0, a1, dash = []) => {
      g.setLineDash(dash);
      g.beginPath();
      g.arc(...this._xy(cx, cz).slice(0, 2), r * s, a0, a1);
      g.stroke();
      g.setLineDash([]);
    };
    arc(0, 5.79, 1.8, 0, Math.PI);
    arc(0, 5.79, 1.8, Math.PI, 2 * Math.PI, [3, 4]);
    arc(0, HOOP_Z, 1.22, 0, Math.PI);
    arc(0, L, 1.8, Math.PI, 2 * Math.PI);
    if (this.play.fullCourt) {
      this._line([[-W / 2, L], [-W / 2, 2 * L], [W / 2, 2 * L], [W / 2, L]]);
      arc(0, L, 1.8, 0, Math.PI);
      this._line([[-2.44, 2 * L], [-2.44, 2 * L - 5.79], [2.44, 2 * L - 5.79], [2.44, 2 * L]]);
      arc(0, 2 * L - HOOP_Z, 6.75, Math.PI + Math.acos(6.6 / 6.75), 2 * Math.PI - Math.acos(6.6 / 6.75));
    }
    const a = Math.acos(6.6 / 6.75);
    this._line([[-6.6, 0], [-6.6, HOOP_Z + 6.75 * Math.sin(a)]]);
    this._line([[6.6, 0], [6.6, HOOP_Z + 6.75 * Math.sin(a)]]);
    arc(0, HOOP_Z, 6.75, a, Math.PI - a);
    // 篮板 + 篮筐
    g.strokeStyle = 'rgba(223, 246, 255, 0.7)';
    g.lineWidth = 2;
    this._line([[-0.9, 1.2], [0.9, 1.2]]);
    g.strokeStyle = rgba(ORANGE, 0.9);
    g.lineWidth = 1.5;
    arc(0, HOOP_Z, 0.23, 0, 2 * Math.PI);
  }

  /** 当前站位最近距离差；不做速度外推，也不代表区域控制。 */
  _positionWash(pos) {
    if (!this.off.length || !this.def.length) return;
    const img = this.heatImg;
    const [gw, gh] = [img.width, img.height];
    const near = (team, x, z) => Math.min(...team.map((p) => Math.hypot(pos[p.id][0] - x, pos[p.id][1] - z)));
    for (let j = 0; j < gh; j++) {
      for (let i = 0; i < gw; i++) {
        const x = -W / 2 + ((i + 0.5) / gw) * W;
        const z = ((j + 0.5) / gh) * L;
        const d = near(this.def, x, z) - near(this.off, x, z);
        const k = 4 * (j * gw + i);
        const c = d >= 0 ? [53, 230, 255] : [255, 154, 46];
        img.data[k] = c[0]; img.data[k + 1] = c[1]; img.data[k + 2] = c[2];
        img.data[k + 3] = Math.round(20 * Math.min(1, Math.abs(d) / 2.5));
      }
    }
    this.hg.putImageData(img, 0, 0);
    const [x, y, scale] = this._xy(-W / 2, 0);
    this.g.save();
    this.g.imageSmoothingEnabled = true;
    this.g.drawImage(this.heat, x, y, W * scale, L * scale);
    this.g.restore();
  }

  _trails(t) {
    const g = this.g;
    const start = Math.max(Math.min(t, this.sourceStart), t - TRAIL, 0);
    for (const p of this.play.players) {
      for (let k = 0; k < 16; k++) {
        const ta = start + (t - start) * k / 16;
        const tb = start + (t - start) * (k + 1) / 16;
        g.strokeStyle = rgba(p.team === 'off' ? CYAN : ORANGE, 0.05 + 0.32 * k / 16);
        g.lineWidth = 1 + k / 16;
        this._line([this.tl.pos(p.id, ta), this.tl.pos(p.id, tb)]);
      }
    }
    // Rebuild from a bounded source-clock sample set. No render-order history,
    // and no repeated samples accumulate during an output-clock freeze.
    if (!this.sampleBall) return;
    const points = [];
    for (let k = Math.ceil((t - start) * BALL_TRAIL_HZ); k >= 0; k--) {
      const b = this.sampleBall(Math.max(start, t - k / BALL_TRAIL_HZ));
      if (b && Number.isFinite(b.x) && Number.isFinite(b.z)) points.push([b.x, b.z]);
    }
    g.strokeStyle = rgba(GOLD, 0.65);
    g.lineWidth = 1.5;
    this._line(points, [3, 3]);
  }

  _arrowHead(x, y, ang, size, color) {
    const g = this.g;
    g.fillStyle = color;
    g.beginPath();
    g.moveTo(x, y);
    g.lineTo(x - size * Math.cos(ang - 0.45), y - size * Math.sin(ang - 0.45));
    g.lineTo(x - size * Math.cos(ang + 0.45), y - size * Math.sin(ang + 0.45));
    g.closePath();
    g.fill();
  }

  /** 战术记法箭头：pts 为屏幕坐标；style = run | dribble | pass。 */
  _notation(pts, style, color) {
    if (pts.length < 2) return;
    const g = this.g;
    g.strokeStyle = color;
    g.lineWidth = 1.6;
    g.lineJoin = 'round';
    g.setLineDash(style === 'pass' ? [5, 4] : []);
    g.beginPath();
    if (style === 'dribble') {                  // 运球：沿路径的折线
      let acc = 0;
      let flip = 1;
      g.moveTo(...pts[0]);
      for (let i = 1; i < pts.length - 1; i++) {
        const [x0, y0] = pts[i - 1];
        const [x1, y1] = pts[i];
        const len = Math.hypot(x1 - x0, y1 - y0) || 1;
        acc += len;
        if (acc > 6) {
          acc = 0;
          flip = -flip;
          g.lineTo(x1 + (-(y1 - y0) / len) * 3.5 * flip, y1 + ((x1 - x0) / len) * 3.5 * flip);
        }
      }
      g.lineTo(...pts.at(-1));
    } else {
      pts.forEach((p, i) => g[i ? 'lineTo' : 'moveTo'](...p));
    }
    g.stroke();
    g.setLineDash([]);
    const [xa, ya] = pts.at(-2);
    const [xb, yb] = pts.at(-1);
    this._arrowHead(xb, yb, Math.atan2(yb - ya, xb - xa), 8, color);
  }

  _annotations(t, holder) {
    for (const a of this.play.annotations ?? []) {
      if (t < a.t[0] || t > a.t[1] || a.type !== 'arrow') continue;
      if (a.path) {
        // Reuse the approved path and crop to the approved reconstruction window.
        const t0 = Math.max(this.sourceStart, a.path.t[0]);
        const t1 = Math.min(this.sourceEnd, a.path.t[1]);
        if (t1 <= t0) continue;
        const pts = Array.from({ length: 25 }, (_, k) => this._xy(...this.tl.pos(a.path.who, t0 + (t1 - t0) * k / 24)).slice(0, 2));
        const dribbling = a.path.who === holder && this.play.ball.some((b) => b.hold === holder && b.mode === 'dribble' && b.t[1] > t0 && b.t[0] < t1);
        this._notation(pts, dribbling ? 'dribble' : 'run', rgba(a.color === 'def' ? ORANGE : CYAN, 0.8));
      } else if (a.from && a.to) {
        const p = this._xy(...this.tl.pos(a.from, t)).slice(0, 2);
        const q = this._xy(...this.tl.pos(a.to, t)).slice(0, 2);
        const d = Math.hypot(q[0] - p[0], q[1] - p[1]);
        if (d <= 24) continue;
        const pad = 12 / d;
        this._notation([[p[0] + (q[0] - p[0]) * pad, p[1] + (q[1] - p[1]) * pad], [q[0] - (q[0] - p[0]) * pad, q[1] - (q[1] - p[1]) * pad]], 'pass', GOLD);
      }
    }
    // Only the existing, explicit screen action gets a conventional T marker.
    for (const p of this.play.players) {
      if (!p.actions?.some(([name, t0, t1]) => name === 'screen' && t >= t0 && t <= t1)) continue;
      const [x, y, s] = this._xy(...this.tl.pos(p.id, t));
      this.g.strokeStyle = GOLD;
      this.g.lineWidth = 2.5;
      this.g.beginPath();
      this.g.moveTo(x - 0.55 * s, y + 0.7 * s);
      this.g.lineTo(x + 0.55 * s, y + 0.7 * s);
      this.g.moveTo(x, y + 0.7 * s);
      this.g.lineTo(x, y + 1.05 * s);
      this.g.stroke();
    }
  }

  _timeline(t) {
    const g = this.g;
    const cw = this.canvas.width;
    const y = this.canvas.height - FOOT + 5;
    const x = (stamp) => 16 + (cw - 32) * Math.min(1, Math.max(0, stamp / this.play.duration));
    for (const [i, ph] of (this.play.phases ?? []).entries()) {
      const active = t >= ph.t[0] && t < ph.t[1];
      g.fillStyle = active ? rgba(GOLD, 0.9) : rgba('#dff6ff', i % 2 ? 0.16 : 0.25);
      g.fillRect(x(ph.t[0]) + 1, y, Math.max(0, x(ph.t[1]) - x(ph.t[0]) - 2), 8);
    }
    g.fillStyle = '#fff';
    g.fillRect(x(t) - 1, y - 4, 2, 16);
    g.font = '16px system-ui,"AdkCJK",sans-serif';
    g.textBaseline = 'middle';
    g.fillStyle = 'rgba(223, 246, 255, 0.7)';
    g.textAlign = 'left';
    g.fillText(`源片 ${t.toFixed(1)} s`, 16, y + 28);
    g.textAlign = 'right';
    g.fillText('轨迹 · 近 2 秒', cw - 16, y + 28);
    g.textAlign = 'left';
    const items = this.rolesVerified
      ? [['●', CYAN, '进攻'], ['▲', ORANGE, '防守'], ['→', CYAN, '跑位'], ['⌇', CYAN, '运球'], ['⇢', GOLD, '传球'], ['⊥', GOLD, '掩护']]
      : [['●', CYAN, this.paletteLabels.off], ['▲', ORANGE, this.paletteLabels.def], ['→', CYAN, '位置尾迹']];
    this.legendTexts = items.map(item=>item[2]);
    const ly = this.canvas.height - 13;
    items.forEach(([symbol, color, label], i) => {
      const lx = 16 + i * (cw - 32) / items.length;
      g.fillStyle = color;
      g.fillText(symbol, lx, ly);
      g.fillStyle = 'rgba(223, 246, 255, 0.7)';
      g.fillText(label, lx + 21, ly);
    });
  }

  update(t, ballPos) {
    const g = this.g;
    g.clearRect(0, 0, this.canvas.width, this.canvas.height);
    // Canvas styles must start from the same state for repeated/乱序 seek.
    g.lineWidth = 1; g.globalAlpha = 1; g.lineJoin = 'miter'; g.lineCap = 'butt'; g.setLineDash([]);
    const pos = Object.fromEntries(this.play.players.map((p) => [p.id, this.tl.pos(p.id, t)]));
    const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
    const holder = this.tl.ballOwner(t);
    const seg = this.play.ball.find((b) => t >= b.t[0] && t <= b.t[1]);
    const inPossession = seg?.hold === holder;

    this._positionWash(pos);
    this._court();
    this._trails(t);

    // 进攻五人连线（站位轮廓）
    const hull = this.rolesVerified ? convexHull(this.off.map((p) => pos[p.id])) : [];
    g.fillStyle = 'rgba(53, 230, 255, 0.03)';
    g.strokeStyle = 'rgba(53, 230, 255, 0.22)';
    g.setLineDash([4, 3]);
    g.beginPath();
    hull.forEach((p, i) => g[i ? 'lineTo' : 'moveTo'](...this._xy(...p)));
    g.closePath();
    g.fill();
    g.stroke();
    g.setLineDash([]);

    // 每个进攻球员 → 最近防守人
    const report = (this.rolesVerified ? this.off : []).map((p) => {
      let best = null;
      for (const d of this.def) {
        const k = dist(pos[p.id], pos[d.id]);
        if (!best || k < best.k) best = { d, k };
      }
      return { p, ...best };
    });
    for (const r of report) {
      const open = r.k > OPEN_DIST && r.p.id !== holder;
      g.strokeStyle = open ? OPEN_COLOR : 'rgba(255,255,255,0.25)';
      g.lineWidth = open ? 1.5 : 1;
      g.beginPath();
      g.moveTo(...this._xy(...pos[r.p.id]));
      g.lineTo(...this._xy(...pos[r.d.id]));
      g.stroke();
      if (open) {
        const [x, y, s] = this._xy(...pos[r.p.id]);
        g.beginPath();
        g.arc(x, y, 0.85 * s, 0, 2 * Math.PI);
        g.stroke();
      }
    }

    this._annotations(t, holder);

    for (const p of this.play.players) {
      const [x, y, s] = this._xy(...pos[p.id]);
      const off = p.team === 'off';
      if (off && p.id === holder && inPossession) {
        g.strokeStyle = GOLD; g.lineWidth = 2;
        g.beginPath(); g.arc(x, y, 0.61 * s, 0, 2 * Math.PI); g.stroke();
      }
      g.fillStyle = off ? CYAN : ORANGE;
      g.globalAlpha = off ? 1 : 0.85;
      g.beginPath();
      if (off) g.arc(x, y, 0.42 * s, 0, 2 * Math.PI);
      else {
        // Fixed orientation is a team symbol, not a measured defensive facing.
        g.moveTo(x, y - 0.57 * s);
        g.lineTo(x - 0.49 * s, y + 0.39 * s);
        g.lineTo(x + 0.49 * s, y + 0.39 * s);
        g.closePath();
      }
      g.fill();
      g.globalAlpha = 1;
      g.fillStyle = '#021016';
      g.font = `bold ${Math.round((off ? 0.42 : 0.36) * s)}px system-ui,"AdkCJK",sans-serif`;
      g.textAlign = 'center';
      g.textBaseline = 'middle';
      g.fillText(String(p.num).replace('X', ''), x, y + 0.5);
    }
    if (ballPos) {
      const [x, y, s] = this._xy(ballPos.x, ballPos.z);
      g.fillStyle = '#ffb347';
      g.strokeStyle = '#fff';
      g.lineWidth = 1;
      g.beginPath();
      g.arc(x, y, 0.2 * s, 0, 2 * Math.PI);
      g.fill();
      g.stroke();
    }

    this._timeline(t);

    if (!this.rolesVerified) {
      // Palette membership supplies neither possession nor defensive roles.
      this.el.phase.textContent = this.el.intent.textContent = this.el.stats.textContent = '';
      this.lastPhase = null; this.lastStats = '';
      return;
    }

    // 阶段 + 指标
    const phase = (this.play.phases ?? []).find((ph) => t >= ph.t[0] && t < ph.t[1]) ?? null;
    if (phase !== this.lastPhase) {
      this.lastPhase = phase;
      this.el.phase.textContent = phase ? phase.name : '';
      this.el.intent.textContent = phase ? phase.intent : '';
    }
    const spacing = mean(this.off.map((p) => Math.min(...this.off.filter((q) => q !== p).map((q) => dist(pos[p.id], pos[q.id])))));
    const chips = [`<span class="chip"><i>进攻平均间距</i><b>${spacing.toFixed(1)} m</b></span>`];
    const rows = [];
    if (holder && inPossession) {
      const h = this.off.find((p) => p.id === holder);
      const r = report.find((x) => x.p.id === holder);
      const n = this.def.filter((d) => dist(pos[holder], pos[d.id]) < PRESSURE_R).length;
      chips.push(`<span class="chip"><i>最近防守距离</i><b>${r.k.toFixed(1)} m</b></span>`, `<span class="chip"><i>${PRESSURE_R} m 内防守</i><b style="color:${n >= 2 ? RED : 'inherit'}">${n} 人</b></span>`);
      rows.push(`<b>持球</b> ${h ? who(h) : holder}`);
    }
    const open = report.filter((r) => r.k > OPEN_DIST && r.p.id !== holder).sort((a, b) => b.k - a.k);
    rows.push(open.length
      ? `<b>空位</b> ${open.map((r) => `<span style="color:${OPEN_COLOR}">${who(r.p)} ${r.k.toFixed(1)} m</span>`).join(' · ')}`
      : '<b>空位</b> 无');
    const html = `<div class="chips">${chips.join('')}</div>${rows.join('<br>')}`;
    if (html !== this.lastStats) {
      this.lastStats = html;
      this.el.stats.innerHTML = html;
    }
  }
}

function mean(a) {
  return a.reduce((s, x) => s + x, 0) / a.length;
}

/** Andrew 单调链凸包，点为 [x, z]。 */
function convexHull(points) {
  const p = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [];
  for (const q of p) {
    while (lower.length >= 2 && cross(lower.at(-2), lower.at(-1), q) <= 0) lower.pop();
    lower.push(q);
  }
  const upper = [];
  for (const q of [...p].reverse()) {
    while (upper.length >= 2 && cross(upper.at(-2), upper.at(-1), q) <= 0) upper.pop();
    upper.push(q);
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}
