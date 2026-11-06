// 篮球：位置是时间的纯函数（BallSolver，拖进度条也正确）+ 真实篮球材质/旋转/落地压缩/地面投影（Ball）。
import * as THREE from 'three';
import { random } from './random.js';
import { COURT } from './court.js';
import { sampleMotion } from './motion.js';

const G = 9.8;
const BALL_R = 0.12;
const DRIBBLE_PERIOD = 0.5;

function poseToWorld(pose, lx, ly, lz) {
  const s = Math.sin(pose.yaw);
  const c = Math.cos(pose.yaw);
  return new THREE.Vector3(pose.x + c * lx + s * lz, ly, pose.z - s * lx + c * lz);
}

/** 落地弹跳：从高度 h0 静止下落，经过 tau 秒后的高度（恢复系数 0.55）。 */
function bounceHeight(h0, tau) {
  let v = 0;
  let y = Math.max(BALL_R, h0);
  let rest = tau;
  for (let k = 0; k < 8 && rest > 0; k++) {
    // 当前弹道：y(t) = y + v t - g t²/2，落到 BALL_R 所需时间
    const disc = v * v + 2 * G * (y - BALL_R);
    const tHit = (v + Math.sqrt(Math.max(0, disc))) / G;
    if (rest < tHit) return y + v * rest - 0.5 * G * rest * rest;
    rest -= tHit;
    v = Math.sqrt(Math.max(0, disc)) * 0.55;
    y = BALL_R;
  }
  return y;
}

// Contact windows share the ball's source clock. The source XYZ branch stays
// verbatim; only the unmeasured procedural dribble is anchored to its retargeted
// palm at each top of the bounce. No frame-to-frame feedback is used.
const CONTACT_HALF_WINDOW = 0.135;
const CONTACT_FULL_WINDOW = 0.045;
const HAND_CORRECTION_LIMIT = 0.18;
const smooth01 = (v) => { const u=Math.min(1,Math.max(0,v)); return u*u*(3-2*u); };
const UP = new THREE.Vector3(0, 1, 0);

export class BallSolver {
  constructor(play, timeline, probe) {
    this.unknown = play.metadata?.ballObservation === 'unknown';
    this.segs = play.ball ?? [];
    if (this.unknown && (this.segs.length || play.ballPath?.length))
      throw new Error('Unknown ball cannot include a replacement path');
    if (!this.unknown && !this.segs.length)
      throw new Error('Empty ball requires explicit unknown observation');
    this.tl = timeline;
    this.probe = probe; // Off-screen rig; it never consumes contacts itself.
    this.motion = play.motion;
    this.players = play.players;
    this.contacts = [];
    this.dribbles = new Map();
    this.gripSides = new Map();
    for (const [k,s] of this.segs.entries()) {
      if (s.hold && s.mode === 'dribble') this._buildDribble(k,s);
      if (s.pass) {
        this._addHandoff(s.pass[0],s.t[0],'pass-release',k);
        this._addHandoff(s.pass[1],s.t[1],'pass-catch',k);
      }
      if (s.shot) this._addHandoff(s.shot,s.t[0],'shot-release',k);
    }
  }

  segIndex(t) {
    if (this.unknown) return -1;
    // A boundary belongs to the following segment, including a pass release.
    const i = this.segs.findLastIndex((s) => s.t[0] <= t);
    return Math.max(0,i);
  }

  handFrames(id,t) {
    if (this.unknown) return null;
    const motion=sampleMotion(this.motion,id,t), pose=this.tl.pose(id,t);
    const player=this.players.find(p=>p.id===id);
    const height=motion?.heightMeters ?? this.motion?.players?.[id]?.heightMeters ?? player?.heightMeters ?? 1.88;
    this.probe.bodyScale=Math.min(1.25,Math.max(.85,height/1.88));
    if(motion)this.probe.setTrackedPose(pose,motion,true);else this.probe.setPose(pose,true);
    const frames={};
    for(const side of ['L','R']) {
      const actual=this.probe.getHandFrame?.(side);
      const wrist=this.probe.joints['hand'+side].clone();
      const along=actual?.along?.clone() ?? wrist.clone().sub(this.probe.joints['elbow'+side]).normalize();
      const palmCenter=actual?.palmCenter?.clone() ?? wrist.clone().addScaledVector(along,.052*this.probe.bodyScale);
      const confidenceIndex=motion?.jointNames?.indexOf(side==='L'?'leftWrist':'rightWrist');
      frames[side]={wrist,palmCenter,along,normal:actual?.normal?.clone() ?? UP.clone(),
        confidence:confidenceIndex>=0?(motion.confidence[confidenceIndex]??0)*(motion.coverageWeight??1):0};
    }
    return {frames,fwd:this.probe.fwd.clone(),right:this.probe.right.clone(),pose};
  }

  contactHandFrame(id,t,side,palmNormal) {
    if (this.unknown) return null;
    const state=this.handFrames(id,t);
    // A procedural top uses the same contact-oriented palm geometry as IK,
    // so a fully extended forearm need not lengthen to rotate its hand.
    const frame=this.probe.getHandFrame?.(side,{palmNormal,weight:1});
    return frame??state.frames[side];
  }

  _pickSide(state,point,previous=null) {
    const score=side=>state.frames[side].palmCenter.distanceTo(point)+.09*(1-state.frames[side].confidence);
    const left=score('L'),right=score('R');
    if(previous && Math.abs(left-right)<.16)return previous;
    return left<right?'L':'R';
  }

  _grip(id,t) {
    const state=this.handFrames(id,t);
    const hold=this.segs.findIndex(s=>s.hold===id&&s.mode!=='dribble'&&t>=s.t[0]&&t<=s.t[1]);
    const key=id+':hold:'+hold;
    let side;
    if(hold>=0) {
      // The selected supporting hand stays fixed over a gather/hold interval;
      // nearest-hand ties cannot make the ball jump between wrists mid-hold.
      if(!this.gripSides.has(key)) {
        const initial=this.handFrames(id,this.segs[hold].t[0]);
        this.gripSides.set(key,this._pickSide(initial,poseToWorld(initial.pose,.32,.9,.4)));
      }
      side=this.gripSides.get(key);
    } else side=this._pickSide(state,poseToWorld(state.pose,.32,.9,.4));
    const frame=this.contactHandFrame(id,t,side,state.fwd);
    return {side,ball:frame.palmCenter.clone().addScaledVector(state.fwd,BALL_R),state};
  }

  _addHandoff(id,time,mode,k) {
    const s=this.segs[k], grip=this._grip(id,time);
    const ball=s.path?.length?this._samplePath(s.path,time):mode.endsWith('release')?this._releaseBall(k):this._catchBall(k);
    this.contacts.push({id,time,side:this._pickSide(grip.state,ball,grip.side),mode,ball,k,
      normal:s.path?.length?null:grip.state.fwd.clone().negate()});
  }

  _buildDribble(k,s) {
    const [t0,t1]=s.t, nodes=[];
    if(s.path?.length) {
      // Detect tops from the supplied ball's vertical reversals. There is no
      // independent cosine phase and no fitting/moving measured XYZ samples.
      const rows=s.path;
      for(let i=1;i<rows.length-1;i++) {
        if(rows[i][2]<=rows[i-1][2]||rows[i][2]<rows[i+1][2])continue;
        if(rows[i][2]-BALL_R<.1)continue;
        const time=rows[i][0];
        if(nodes.length && time-nodes.at(-1).time<.22)continue;
        const ball=new THREE.Vector3(...rows[i].slice(1)),state=this.handFrames(s.hold,time);
        nodes.push({id:s.hold,time,side:this._pickSide(state,ball,nodes.at(-1)?.side),mode:'dribble',ball,k,normal:UP.clone()});
      }
    } else {
      // A full bounce is retained between every pair of palm contacts. Length
      // comes from this possession segment, so both handoffs occur at a top.
      const count=Math.max(1,Math.round((t1-t0)/DRIBBLE_PERIOD));
      let previous=null;
      for(let i=0;i<=count;i++) {
        const time=t0+(t1-t0)*i/count,state=this.handFrames(s.hold,time);
        const nominal=poseToWorld(state.pose,.32,.95,.4);
        let side=this._pickSide(state,nominal,previous);
        let ball=this.contactHandFrame(s.hold,time,side,UP.clone().negate()).palmCenter.clone().addScaledVector(UP,-BALL_R);
        if(i===0 || i===count) {
          const grip=this._grip(s.hold,time);side=grip.side;ball=grip.ball;
        }
        // These are procedural anchors, not corrected source-ball observations.
        // Only a physically supported low dribble is allowed near the court.
        if(i===0&&this.segs[k-1]?.path?.length)ball=this._samplePath(this.segs[k-1].path,time);
        if(i===count&&this.segs[k+1]?.path?.length)ball=this._samplePath(this.segs[k+1].path,time);
        if(!((i===0&&this.segs[k-1]?.path?.length)||(i===count&&this.segs[k+1]?.path?.length)))ball.y=Math.max(BALL_R+.08,ball.y);
        nodes.push({id:s.hold,time,side,mode:'dribble',ball,k,
          normal:i===0||i===count?state.fwd.clone().negate():UP.clone()});
        previous=side;
      }
      for(const node of nodes) {
        const frameAt=dt=> {
          const normal=node.normal??UP;
          const frame=this.contactHandFrame(s.hold,node.time+dt,node.side,normal.clone().negate());
          return frame.palmCenter.clone().addScaledVector(normal,-BALL_R);
        };
        node.velocity=frameAt(.0125).sub(frameAt(-.0125)).multiplyScalar(40);
        node.velocity.y=0;
      }
      this.dribbles.set(k,nodes);
    }
    this.contacts.push(...nodes);
  }

  _releaseBall(k) {
    const s=this.segs[k];
    return k>0?this.inSeg(k-1,s.t[0]):this._grip(s.pass?.[0]??s.shot,s.t[0]).ball;
  }

  _catchBall(k) {
    const s=this.segs[k],next=this.segs[k+1];
    return next?.path?.length?this._samplePath(next.path,s.t[1]):this._grip(s.pass[1],s.t[1]).ball;
  }

  _samplePath(rows,t) {
    if(t<=rows[0][0])return new THREE.Vector3(...rows[0].slice(1));
    if(t>=rows.at(-1)[0])return new THREE.Vector3(...rows.at(-1).slice(1));
    let lo=0,hi=rows.length-1;
    while(lo+1<hi){const mid=(lo+hi)>>1;if(rows[mid][0]<=t)lo=mid;else hi=mid;}
    const a=rows[lo],b=rows[hi];
    return new THREE.Vector3(...a.slice(1)).lerp(new THREE.Vector3(...b.slice(1)),(t-a[0])/(b[0]-a[0]));
  }

  _dribbleAt(k,t) {
    const nodes=this.dribbles.get(k);
    if(t<=nodes[0].time)return nodes[0].ball.clone();
    if(t>=nodes.at(-1).time)return nodes.at(-1).ball.clone();
    let i=0;while(nodes[i+1].time<t)i++;
    const a=nodes[i],b=nodes[i+1],span=b.time-a.time,u=(t-a.time)/span,w=smooth01(u);
    const linear=a.ball.clone().lerp(b.ball,u);
    const p=a.ball.clone().multiplyScalar(2*u**3-3*u*u+1)
      .addScaledVector(a.velocity,(u**3-2*u*u+u)*span)
      .addScaledVector(b.ball,-2*u**3+3*u*u)
      .addScaledVector(b.velocity,(u**3-u*u)*span);
    // Keep curvature close to the interval anchors while preserving the hand's
    // horizontal tangent at contact; a moving handler need not stop each bounce.
    const delta=p.clone().sub(linear);delta.y=0;const distance=delta.length();
    const factor=distance>0?.08*Math.tanh(distance/.08)/distance:1;
    p.copy(linear).addScaledVector(delta,factor);
    // Vertical flight remains free away from the two short palm windows, and
    // reaches the floor exactly halfway through each cycle.
    const top=THREE.MathUtils.lerp(a.ball.y,b.ball.y,w);
    p.y=BALL_R+(top-BALL_R)*Math.abs(Math.cos(Math.PI*u));
    return p;
  }

  _gatherAt(k,t) {
    const s=this.segs[k],prev=this.segs[k-1],next=this.segs[k+1];
    let ball=this._grip(s.hold,t).ball;
    // Mixed source/procedural segments retain supplied boundary positions. A
    // conflicting hand is gated by contactsAt rather than moving source XYZ.
    if(prev?.path?.length&&t<s.t[0]+.18)ball=this._samplePath(prev.path,s.t[0]).lerp(ball,smooth01((t-s.t[0])/.18));
    if(next?.path?.length&&t>s.t[1]-.18)ball.lerp(this._samplePath(next.path,s.t[1]),smooth01((t-(s.t[1]-.18))/.18));
    return ball;
  }

  handsMid(id,t) {
    if (this.unknown) return null;
    const {frames,fwd}=this.handFrames(id,t);
    return frames.L.palmCenter.clone().add(frames.R.palmCenter).multiplyScalar(.5).addScaledVector(fwd,.1);
  }

  contactsAt(id,t) {
    if (this.hiddenAt(t)) return [];
    const events=this.contacts.filter(e=>e.id===id&&Math.abs(t-e.time)<=CONTACT_HALF_WINDOW);
    const s=this.segs[this.segIndex(t)];
    if(s.hold===id&&s.mode!=='dribble') {
      const grip=this._grip(id,t);
      events.push({id,time:t,side:grip.side,mode:'gather',ball:grip.ball,k:this.segIndex(t),normal:grip.state.fwd.clone().negate()});
    }
    if(!events.length)return [];
    const state=this.handFrames(id,t),ball=this.at(t),bySide=new Map();
    for(const event of [...events]) {
      if(event.mode==='dribble')continue;
      const side=event.side==='L'?'R':'L',palm=state.frames[side].palmCenter;
      const distance=palm.distanceTo(ball);
      // A nearby guide/catching hand may share the gather. Far-away observed
      // hands are left alone instead of manufacturing an impossible grip.
      if(Math.abs(distance-BALL_R)<=HAND_CORRECTION_LIMIT)events.push({...event,side,
        normal:palm.clone().sub(ball).normalize(),mode:event.mode+'-support',
        supportWeight:1-smooth01((Math.abs(distance-BALL_R)-.10)/.08)});
    }
    for(const event of events) {
      const dt=Math.abs(t-event.time);
      let weight=1-smooth01((dt-CONTACT_FULL_WINDOW)/(CONTACT_HALF_WINDOW-CONTACT_FULL_WINDOW));
      // Catch contact builds as the incoming ball arrives; after release the
      // hand lets go. A symmetric window would glue palms to free-flight XYZ.
      if(s.pass&&s.pass[1]===id&&event.time===s.t[1]&&t<event.time)weight*=smooth01(1+(t-event.time)/CONTACT_FULL_WINDOW);
      if((s.pass?.[0]===id||s.shot===id)&&event.time===s.t[0]&&t>event.time)weight*=1-smooth01((t-event.time)/CONTACT_FULL_WINDOW);
      const frame=state.frames[event.side];
      // Once a pass/shot leaves the hand, its release constraint fades at
      // that endpoint. Following the fast free-flight ball for even 45 ms
      // would pull the wrist forward and then snap it back on gate removal.
      const released=(s.pass?.[0]===id||s.shot===id)&&event.time===s.t[0]&&t>event.time;
      const contactBall=released?event.ball:ball;
      // A source/path conflict does not grant permission to drag a wrist across
      // the court. The body IK applies this same finite correction limit.
      const normal=event.normal ?? frame.palmCenter.clone().sub(contactBall).normalize();
      if(normal.lengthSq()<.5)normal.copy(UP);
      const palmTarget=contactBall.clone().addScaledVector(normal,BALL_R);
      const target=palmTarget.clone().sub(frame.palmCenter.clone().sub(frame.wrist));
      const gap=target.distanceTo(frame.wrist);
      const reachGate=1-smooth01((gap-HAND_CORRECTION_LIMIT)/.15);
      const payload={side:event.side,target,weight:weight*reachGate*(event.supportWeight??1),
        maxCorrectionMeters:HAND_CORRECTION_LIMIT,mode:event.mode,time:t,
        ball:contactBall.clone(),releaseTail:released,ballRadius:BALL_R,palmTarget,palmNormal:normal.clone().negate(),
        provenance:this.segs[event.k].path?.length?'source-ball-clock':'procedural-ball-clock'};
      if(payload.weight>0 && (!bySide.has(event.side)||bySide.get(event.side).weight<payload.weight))bySide.set(event.side,payload);
    }
    return [...bySide.values()];
  }

  /** Ball not shown: before the first segment, or inside a {hidden:true} segment (possession unknown). */
  hiddenAt(t) {
    if (this.unknown) return true;
    return t < this.segs[0].t[0] || Boolean(this.segs[this.segIndex(t)].hidden);
  }

  at(t) { return this.hiddenAt(t) ? null : this.inSeg(this.segIndex(t),t); }

  inSeg(k,t) {
    if (this.unknown) return null;
    const s=this.segs[Math.max(0,Math.min(this.segs.length-1,k))];
    if(s.path?.length)return this._samplePath(s.path,t);
    const [t0,t1]=s.t,tau=Math.min(1,Math.max(0,(t-t0)/((t1-t0)||1e-6)));
    if(s.hold&&s.mode==='dribble')return this._dribbleAt(k,t);
    if(s.hold)return this._gatherAt(k,t);
    if(s.pass||s.shot) {
      const a=this._releaseBall(k);
      const b=s.pass?this._catchBall(k):COURT.hoop.clone();
      const y=a.y+(b.y-a.y)*tau+4*(s.arc??.5)*tau*(1-tau);
      return a.lerp(b,tau).setY(y);
    }
    const dt=Math.max(0,t-t0),drift=.5*(1-Math.exp(-dt*1.2));
    const anchor=k>0?this.inSeg(k-1,t0):COURT.hoop;
    return new THREE.Vector3(anchor.x+drift*.3,bounceHeight(anchor.y,dt),anchor.z+drift);
  }
}

/** 程序化篮球贴图：橙色颗粒表皮 + 8 条黑色缝线（赤道、经线、两条弯缝）。 */
function basketballTexture() {
  const W = 1024;
  const H = 512;
  const cv = document.createElement('canvas');
  cv.width = W;
  cv.height = H;
  const ctx = cv.getContext('2d');
  const img = ctx.createImageData(W, H);
  const seam = 0.022;
  for (let j = 0; j < H; j++) {
    const lat = Math.PI * (0.5 - (j + 0.5) / H);
    for (let i = 0; i < W; i++) {
      const lon = 2 * Math.PI * ((i + 0.5) / W);
      const x = Math.cos(lat) * Math.cos(lon);
      const y = Math.sin(lat);
      const z = Math.cos(lat) * Math.sin(lon);
      const onSeam = Math.abs(y) < seam || Math.abs(z) < seam || Math.abs(Math.abs(x) - 0.7) < seam * 0.9;
      const grain = 0.88 + 0.12 * random();
      const k = (j * W + i) * 4;
      if (onSeam) img.data.set([22, 12, 8, 255], k);
      else img.data.set([232 * grain, 112 * grain, 38 * grain, 255], k);
    }
  }
  ctx.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

export class Ball {
  constructor() {
    this.group = new THREE.Group();
    const tex = basketballTexture();
    this.mesh = new THREE.Mesh(new THREE.SphereGeometry(BALL_R, 48, 32), new THREE.MeshStandardMaterial({
      map: tex, emissiveMap: tex, emissive: 0xffffff, emissiveIntensity: 0.35, roughness: 0.75, metalness: 0,
    }));
    this.group.add(this.mesh);
    this.rimLight = new THREE.PointLight(0xffa860, 0.6, 1.5);
    this.group.add(this.rimLight);

    // 地面投影：随高度变大变淡，读高度用
    this.shadow = new THREE.Mesh(new THREE.CircleGeometry(BALL_R, 32), new THREE.MeshBasicMaterial({
      color: 0xffa040, transparent: true, opacity: 0.5, depthWrite: false, blending: THREE.AdditiveBlending }));
    this.shadow.rotation.x = -Math.PI / 2;

    this.trailN = 20;
    this.trailPos = new Float32Array(this.trailN * 3);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(this.trailPos, 3));
    this.trail = new THREE.Points(g, new THREE.PointsMaterial({
      color: 0xffb070, size: 0.045, transparent: true, opacity: 0.45, depthWrite: false, blending: THREE.AdditiveBlending }));
    this.trail.frustumCulled = false;
    this.lastT = null;
  }

  update(solver, t) {
    const p = solver.at(t);
    if (p === null) {
      this.group.visible = this.mesh.visible = this.shadow.visible = this.trail.visible = false;
      this.rimLight.visible = false;
      this.lastT = null;
      return;
    }
    this.group.visible = this.mesh.visible = this.shadow.visible = this.rimLight.visible = true;
    const prev = solver.at(Math.max(0, t - 0.01)) ?? p;
    const v = p.clone().sub(prev).multiplyScalar(100);
    this.group.position.copy(p);

    // Absolute-time orientation: seeking/render order cannot change export frames.
    const seg = solver.segs[solver.segIndex(t)];
    const spin = seg.pass || seg.shot ? -5 * Math.PI * t : 8 * t;
    this.mesh.rotation.set(spin, 0.22 * t, 0.12 * Math.sin(t));
    this.lastT = t;

    // 触地瞬间压扁一点（运球/落地）
    const squash = Math.max(0, 1 - (p.y - BALL_R) / 0.05) * 0.12;
    this.mesh.scale.set(1 + squash * 0.5, 1 - squash, 1 + squash * 0.5);
    this.mesh.position.y = -squash * BALL_R;

    const h = Math.max(0, p.y - BALL_R);
    this.shadow.position.set(p.x, 0.008, p.z);
    this.shadow.scale.setScalar(1 + h * 0.6);
    this.shadow.material.opacity = 0.45 / (1 + h * 1.5);

    for (let i = 0; i < this.trailN; i++) {
      const q = solver.at(Math.max(0, t - (i + 1) * 0.02)) ?? p;   // trail stops at a hidden interval
      this.trailPos.set([q.x, q.y, q.z], i * 3);
    }
    this.trail.geometry.attributes.position.needsUpdate = true;
    this.trail.visible = Boolean(seg.pass || seg.shot) && v.length() > 3; // Free flight only; a moving handler is not a pass.
  }
}
