// 把战术脚本预计算成 120Hz 查找表（位置/朝向/速度/步态相位），任意时刻可确定性取样 → 支持拖动进度条。
import { COURT } from './court.js';

const HZ = 120;
const RAMP = 0.25;          // 动作淡入淡出（秒）
const STRIDE_LEN = 2.3;     // 一个完整步态周期的距离（米）
const RUN_SPEED = 5.0;      // 达到满跑姿态的速度（米/秒）

const clamp01 = (x) => Math.min(1, Math.max(0, x));
const smooth = (x) => x * x * (3 - 2 * x);

/** 非均匀时间 Catmull-Rom（Hermite）插值。 */
function samplePath(path, t) {
  if (t <= path[0][0]) return [path[0][1], path[0][2]];
  const n = path.length;
  if (t >= path[n - 1][0]) return [path[n - 1][1], path[n - 1][2]];
  let i = 0;
  while (path[i + 1][0] < t) i++;
  const p0 = path[Math.max(0, i - 1)];
  const p1 = path[i];
  const p2 = path[i + 1];
  const p3 = path[Math.min(n - 1, i + 2)];
  const h = (p2[0] - p1[0]) || 1e-6;
  const s = (t - p1[0]) / h;
  const h00 = 2 * s ** 3 - 3 * s ** 2 + 1;
  const h10 = s ** 3 - 2 * s ** 2 + s;
  const h01 = -2 * s ** 3 + 3 * s ** 2;
  const h11 = s ** 3 - s ** 2;
  // 关键帧静止段（相邻两帧同位置）切线置零，避免过冲
  const same = (a, b) => a[1] === b[1] && a[2] === b[2];
  const tan = (a, b, k) => (a[0] === b[0] ? 0 : (b[k] - a[k]) / (b[0] - a[0]));
  const still1 = same(p1, p2);
  return [1, 2].map((k) => {
    const m1 = still1 || (i > 0 && same(p0, p1)) ? 0 : tan(p0, p2, k);
    const m2 = still1 || (i + 2 < n && same(p2, p3)) ? 0 : tan(p1, p3, k);
    return h00 * p1[k] + h10 * h * m1 + h01 * p2[k] + h11 * h * m2;
  });
}

export function actionWeight(actions, name, t) {
  let w = 0;
  for (const [a, t0, t1] of actions) {
    if (a !== name || t < t0 - RAMP || t > t1 + RAMP) continue;
    w = Math.max(w, smooth(clamp01(Math.min((t - t0 + RAMP) / RAMP, (t1 + RAMP - t) / RAMP))));
  }
  return w;
}

function angleLerp(a, b, w) {
  let d = b - a;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return a + d * w;
}

export class Timeline {
  constructor(play) {
    this.play = play;
    this.duration = play.duration;
    this.n = Math.ceil(play.duration * HZ) + 1;
    this.byId = {};
    for (const p of play.players) {
      const tb = { x: new Float32Array(this.n), z: new Float32Array(this.n), vx: new Float32Array(this.n),
        vz: new Float32Array(this.n), dist: new Float32Array(this.n), yaw: new Float32Array(this.n) };
      for (let i = 0; i < this.n; i++) [tb.x[i], tb.z[i]] = samplePath(p.path, i / HZ);
      for (let i = 0; i < this.n; i++) {
        const a = Math.max(0, i - 2);
        const b = Math.min(this.n - 1, i + 2);
        tb.vx[i] = ((tb.x[b] - tb.x[a]) * HZ) / (b - a);
        tb.vz[i] = ((tb.z[b] - tb.z[a]) * HZ) / (b - a);
        if (i > 0) tb.dist[i] = tb.dist[i - 1] + Math.hypot(tb.x[i] - tb.x[i - 1], tb.z[i] - tb.z[i - 1]);
      }
      this.byId[p.id] = { def: p, tb };
    }
    // 朝向依赖球的位置，所以在全部轨迹就绪后再算，并做指数平滑
    for (const { def, tb } of Object.values(this.byId)) {
      let yaw = null;
      for (let i = 0; i < this.n; i++) {
        const t = i / HZ;
        const speed = Math.hypot(tb.vx[i], tb.vz[i]);
        const [fx, fz] = this._faceTarget(def, t);
        const faceYaw = Math.atan2(fx - tb.x[i], fz - tb.z[i]);
        const moveYaw = Math.atan2(tb.vx[i], tb.vz[i]);
        const defending = actionWeight(def.actions, 'defend', t);
        const m = clamp01((speed - 0.5) / 1.5) * (1 - 0.85 * defending);
        const target = angleLerp(faceYaw, moveYaw, m);
        yaw = yaw === null ? target : angleLerp(yaw, target, 0.08);
        tb.yaw[i] = yaw;
      }
    }
  }

  _faceTarget(def, t) {
    if (def.face === 'hoop' && actionWeight(def.actions, 'dribble', t) < 0.5) return [COURT.hoop.x, COURT.hoop.z];
    const holder = this.ballOwner(t);
    if (holder && holder !== def.id) return this.pos(holder, t);
    if (def.face === 'ball' && holder === def.id) return [COURT.hoop.x, COURT.hoop.z];
    // Ball unknown: a defender faces his matchup, everyone else the hoop.
    if (!holder && def.mark && this.byId[def.mark]) return this.pos(def.mark, t);
    return [COURT.hoop.x, COURT.hoop.z];
  }

  /** 当前持球/传球目标的球员 id（用于朝向）。 */
  ballOwner(t) {
    for (const seg of this.play.ball) {
      if (t < seg.t[0] || t > seg.t[1]) continue;
      if (seg.hold) return seg.hold;
      if (seg.pass) return seg.pass[1];
      if (seg.shot) return seg.shot;
    }
    return null;
  }

  idx(t) {
    return Math.min(this.n - 1, Math.max(0, Math.round(t * HZ)));
  }

  value(tb, name, t) {
    const position=Math.min(this.n-1,Math.max(0,t*HZ));
    const first=Math.floor(position),second=Math.min(this.n-1,first+1),amount=position-first;
    return name==='yaw'?angleLerp(tb[name][first],tb[name][second],amount):
      tb[name][first]+(tb[name][second]-tb[name][first])*amount;
  }

  pos(id, t) {
    const { tb } = this.byId[id];
    return [this.value(tb,'x',t),this.value(tb,'z',t)];
  }

  /** 给 HoloPlayer.setPose 用的完整姿态（不含运球手目标，由 scene 补）。 */
  pose(id, t) {
    const { def, tb } = this.byId[id];
    const vx=this.value(tb,'vx',t),vz=this.value(tb,'vz',t);
    const speed = Math.hypot(vx,vz);
    const yaw = this.value(tb,'yaw',t);
    // 世界速度方向 → 身体局部方向
    const lx = speed > 1e-3 ? (vx * Math.cos(yaw) - vz * Math.sin(yaw)) / speed : 0;
    const lz = speed > 1e-3 ? (vx * Math.sin(yaw) + vz * Math.cos(yaw)) / speed : 1;
    const w = (name) => actionWeight(def.actions, name, t);
    const defend = w('defend');
    const screen = w('screen');
    const dribble = w('dribble');
    const jump = def.actions.filter(([a, t0, t1]) => a === 'jump' && t >= t0 && t <= t1)
      .reduce((acc, [, t0, t1]) => Math.max(acc, Math.sin((Math.PI * (t - t0)) / ((t1 - t0) || 1e-6)) * 0.55), 0);
    return {
      x: this.value(tb,'x',t), z: this.value(tb,'z',t), yaw,
      phase: (this.value(tb,'dist',t) / STRIDE_LEN) * Math.PI * 2,
      speed: clamp01(speed / RUN_SPEED) * (1 - 0.4 * defend),
      strideX: lx, strideZ: lz,
      crouch: Math.max(0.85 * defend, 0.35 * screen, 0.3 * dribble) * (1 - clamp01(jump * 3)),
      guard: 0.95 * defend, screen, dribble, catchBall: w('catch'), shoot: w('shoot'), jump,
    };
  }
}

export { samplePath, HZ };
