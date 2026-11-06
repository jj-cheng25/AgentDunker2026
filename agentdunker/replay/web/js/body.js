// Continuous anatomical envelopes, driven by the existing 17-joint rig.
// The torso is one deforming skin; two-bone limbs retain rigid muscle volumes
// away from a small rounded hinge. Hand/face detail is designed, not measured.
import * as THREE from 'three';
import { random } from './random.js';
const UP = new THREE.Vector3(0, 1, 0);
const GOLDEN = Math.PI * (3 - Math.sqrt(5));
const DENSITY = 3900;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const smooth = (v) => { const t=clamp(v,0,1); return t*t*(3-2*t); };
// C1 contact limit: it preserves the surface above the contact band and bends
// designed hand detail onto the floor without moving observed wrist joints.
function contactHeight(value,floor,width){
  const h=value-floor;
  if(h>=width)return [value,1];if(h<=-width)return [floor,0];
  return [floor+(h+width)**2/(4*width),(h+width)/(2*width)];
}
function profileAt(prof, t) {
  const x=clamp(t,0,1)*(prof.length-1), i=Math.min(prof.length-2,Math.floor(x));
  return THREE.MathUtils.lerp(prof[i],prof[i+1],smooth(x-i));
}
function solveIK(a, t, l1, l2, pole, out) {
  const d=new THREE.Vector3().subVectors(t,a);
  const requested=d.length();
  const dist=clamp(requested,Math.abs(l1-l2)+1e-4,l1+l2-1e-3);
  if(requested<1e-8)d.set(0,-1,0);else d.divideScalar(requested);
  const along=(l1*l1-l2*l2+dist*dist)/(2*dist), h=Math.sqrt(Math.max(0,l1*l1-along*along));
  const bend=pole.clone().addScaledVector(d,-pole.dot(d));
  if(bend.lengthSq()<1e-8)bend.crossVectors(d,UP);
  if(bend.lengthSq()<1e-8)bend.set(1,0,0);bend.normalize();
  out.copy(a).addScaledVector(d,along).addScaledVector(bend,h);t.copy(a).addScaledVector(d,dist);return out;
}
// Each part is a continuous swept surface, not a stack of joint spheres.
const PARTS=[
  {name:'torso',rings:40,sides:28,len:.72,radius:.15},
  {name:'neck',rings:12,sides:18,len:.12,radius:.052},
  {name:'head',rings:28,sides:28,len:.23,radius:.08},
  ...['L','R'].flatMap(side=>[
    {name:'arm'+side,side,rings:48,sides:20,len:.61,radius:.056},
    {name:'leg'+side,side,rings:52,sides:22,len:.95,radius:.081},
    {name:'palm'+side,side,rings:14,sides:18,len:.098,radius:.03},
    {name:'shoe'+side,side,rings:26,sides:20,len:.28,radius:.05},
    ...[0,1,2,3,4].map(finger=>({name:'finger'+side+finger,side,finger,rings:12,sides:10,len:finger===4?.07:.074,radius:finger===4?.009:.0075})),
  ]),
];
const STRIDE=17; // centre, lateral axis, depth axis, tangent, radii, two radius slopes, u
function splineAt(knots,u) {
  let i=0;while(i<knots.length-2&&knots[i+1][0]<u)i++;
  const [t0,p0]=knots[i], [t1,p1]=knots[i+1],h=t1-t0,s=clamp((u-t0)/h,0,1);
  const prev=knots[Math.max(0,i-1)],next=knots[Math.min(knots.length-1,i+2)];
  const m0=new THREE.Vector3().subVectors(p1,prev[1]).divideScalar(t1-prev[0]||1);
  const m1=new THREE.Vector3().subVectors(next[1],p0).divideScalar(next[0]-t0||1);
  const c=p0.clone().multiplyScalar(2*s**3-3*s*s+1).addScaledVector(m0,(s**3-2*s*s+s)*h)
    .addScaledVector(p1,-2*s**3+3*s*s).addScaledVector(m1,(s**3-s*s)*h);
  const tangent=p0.clone().multiplyScalar((6*s*s-6*s)/h).addScaledVector(m0,3*s*s-4*s+1)
    .addScaledVector(p1,(-6*s*s+6*s)/h).addScaledVector(m1,3*s*s-2*s);
  if(tangent.lengthSq()<1e-8)tangent.subVectors(p1,p0);tangent.normalize();return {c,tangent};
}
// Only the immediate hinge bends. The upper/lower limb centres stay on their
// measured bone axes; this prevents an arm or shin from becoming a soft hose.
function hingeAt(a,b,c,u,r) {
  const da=new THREE.Vector3().subVectors(b,a),db=new THREE.Vector3().subVectors(c,b);
  const la=da.length(),lb=db.length(),total=la+lb;da.divideScalar(la||1);db.divideScalar(lb||1);
  const d=u<=.5?u*2*la:la+(u-.5)*2*lb,corner=Math.min(r,la*.2,lb*.2),start=la-corner,end=la+corner;
  let centre,tangent;
  if(d<start){centre=a.clone().addScaledVector(da,d);tangent=da;}
  else if(d>end){centre=b.clone().addScaledVector(db,d-la);tangent=db;}
  else {const q=clamp((d-start)/(2*corner||1),0,1),p=b.clone().addScaledVector(da,-corner),n=b.clone().addScaledVector(db,corner);
    centre=p.multiplyScalar((1-q)**2).addScaledVector(b,2*q*(1-q)).addScaledVector(n,q*q);
    tangent=da.clone().multiplyScalar(1-q).addScaledVector(db,q).normalize();}
  return {c:centre,tangent,local:d<=la?d/(la||1):(d-la)/(lb||1),upper:d<=la,hinge:.5,bend:Math.acos(clamp(da.dot(db),-1,1))};
}
function unit(vector,fallback=UP){return vector.lengthSq()>1e-12?vector.normalize():vector.copy(fallback).normalize();}
function transverse(tangent,preferred,backup){
 const result=preferred.clone().addScaledVector(tangent,-preferred.dot(tangent));
 if(result.lengthSq()<1e-8){result.copy(backup??UP).addScaledVector(tangent,-(backup??UP).dot(tangent));}
 if(result.lengthSq()<1e-8){result.set(0,0,1).addScaledVector(tangent,-tangent.z);}
 return unit(result,new THREE.Vector3(1,0,0));
}
function transport(lateral,from,to){
 const a=unit(from.clone()),b=unit(to.clone()),dot=clamp(a.dot(b),-1,1);let q;
 if(dot<-.999999)q=new THREE.Quaternion().setFromAxisAngle(transverse(a,lateral),Math.PI);
 else q=new THREE.Quaternion().setFromUnitVectors(a,b);
 return transverse(b,lateral.clone().applyQuaternion(q));
}
function vectorOf(value){if(value?.isVector3)return value.clone();if(Array.isArray(value)&&value.length===3&&value.every(Number.isFinite))return new THREE.Vector3(...value);return null;}
function relativeAngle(a,b){return Math.atan2(Math.sin(a-b),Math.cos(a-b));}

function putRing(data,i,c,tangent,lateral,rx,rz,u) {
  tangent=unit(tangent.clone());
  const x=transverse(tangent,lateral);
  const z=new THREE.Vector3().crossVectors(x,tangent).normalize(), k=i*STRIDE;
  data.set([c.x,c.y,c.z,x.x,x.y,x.z,z.x,z.y,z.z,tangent.x,tangent.y,tangent.z,rx,rz,0,0,u],k);
}
function finishRings(part){
  const d=part.frames;
  for(let i=0;i<=part.rings;i++){const a=Math.max(0,i-1)*STRIDE,b=Math.min(part.rings,i+1)*STRIDE,k=i*STRIDE;
    const len=Math.hypot(d[b]-d[a],d[b+1]-d[a+1],d[b+2]-d[a+2])||1e-6;
    d[k+14]=clamp((d[b+12]-d[a+12])/len,-6,6);d[k+15]=clamp((d[b+13]-d[a+13])/len,-6,6);
  }
}
const POINT_VS=/* glsl */`
 attribute vec3 nrm; attribute float vis; uniform float uTime,uSize,uScale; varying float vAlpha;
 void main(){vec3 V=normalize(cameraPosition-position);float ndv=dot(nrm,V);float rim=1.0-abs(ndv);
 float front=ndv>0.0?1.0:0.34;float scan=0.94+0.06*sin(position.y*95.0-uTime*1.4);
 vAlpha=(0.31+0.86*pow(rim,1.55))*front*scan*vis;vec4 mv=modelViewMatrix*vec4(position,1.0);
 gl_PointSize=clamp(uSize*uScale/-mv.z,1.1,6.0);gl_Position=projectionMatrix*mv;}`;
const POINT_FS=/* glsl */`
 uniform vec3 uColor;uniform float uOpacity;varying float vAlpha;
 void main(){float d=length(gl_PointCoord-0.5);if(d>0.5)discard;
 gl_FragColor=vec4(uColor*(1.21-d*.55),(1.0-d*1.8)*vAlpha*uOpacity);}`;
// A faint anatomical surface gives the cloud coherent volume at wide angles.
// It shares the same moving envelope and seam suppression as the particles.
const SKIN_VS=/* glsl */`
 attribute float vis; varying vec3 vN,vP; varying float vVis;
 void main(){vN=normal;vP=position;vVis=vis;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}`;
const SKIN_FS=/* glsl */`
 uniform vec3 uColor; uniform float uTime,uOpacity;varying vec3 vN,vP;varying float vVis;
 void main(){if(vVis<0.18)discard;vec3 n=normalize(vN),v=normalize(cameraPosition-vP);
 float d=dot(n,v),rim=pow(1.0-abs(d),2.0),light=0.5+0.5*max(0.0,dot(n,normalize(vec3(-0.4,0.8,0.7))));
 float front=d>0.0?1.0:0.3;float scan=.92+.08*sin(vP.y*95.0-uTime*1.4);
 gl_FragColor=vec4(uColor*(.62+.38*light),(.030+.080*rim)*front*scan*vVis*uOpacity);}`;
const DISC_FS = /* glsl */ `
  uniform vec3 uColor;
  uniform float uTime, uOpacity;
  varying vec2 vUv;
  void main() {
    vec2 p = vUv * 2.0 - 1.0;
    float r = length(p);
    if (r > 1.0) discard;
    float fill = 0.1 + 0.32 * r * r;
    float rim = smoothstep(0.86, 0.97, r) * (1.0 - smoothstep(0.97, 1.0, r));
    float inner = smoothstep(0.6, 0.62, r) * (1.0 - smoothstep(0.62, 0.64, r)) * 0.5;
    float a = uTime * 0.8;
    float sweep = (1.0 - smoothstep(0.0, 0.025, abs(dot(p, vec2(-sin(a), cos(a)))))) * 0.6;
    float alpha = (fill + rim * 1.2 + inner + sweep) * uOpacity * 0.6;
    gl_FragColor = vec4(uColor * (0.7 + rim), alpha);
  }`;
const DISC_VS = /* glsl */ `
  varying vec2 vUv;
  void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;

export class HoloPlayer {
  constructor(color,shared,heightMeters=1.88){
    this.color=new THREE.Color(color);this.joints={};this.bodyScale=Number.isFinite(heightMeters)?clamp(heightMeters/1.88,.85,1.25):1;
    this.parts=PARTS.map(p=>({...p,frames:new Float32Array((p.rings+1)*STRIDE)}));
    this.template=[];this.meshTemplate=[];const indices=[];
    this.parts.forEach((part,pi)=>{
      const count=Math.round(2*Math.PI*part.radius*part.len*DENSITY);
      for(let n=0;n<count;n++){const u=(n+.5+(random()-.5)*.55)/count,a=n*GOLDEN;
        this.template.push([pi,u,Math.cos(a),Math.sin(a)]);}
      const start=this.meshTemplate.length;
      for(let r=0;r<=part.rings;r++)for(let a=0;a<=part.sides;a++){
        const theta=a/part.sides*Math.PI*2;this.meshTemplate.push([pi,r/part.rings,Math.cos(theta),Math.sin(theta)]);}
      for(let r=0;r<part.rings;r++)for(let a=0;a<part.sides;a++){
        const k=start+r*(part.sides+1)+a;indices.push(k,k+1,k+part.sides+1,k+1,k+part.sides+2,k+part.sides+1);}
    });
    const count=this.template.length;this.tplData=Float32Array.from(this.template.flat());this.meshTplData=Float32Array.from(this.meshTemplate.flat());
    this.positions=new Float32Array(count*3);this.normals=new Float32Array(count*3);this.vis=new Float32Array(count).fill(1);
    this.meshPositions=new Float32Array(this.meshTemplate.length*3);this.meshNormals=new Float32Array(this.meshTemplate.length*3);this.meshVis=new Float32Array(this.meshTemplate.length).fill(1);
    const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.BufferAttribute(this.positions,3));geo.setAttribute('nrm',new THREE.BufferAttribute(this.normals,3));geo.setAttribute('vis',new THREE.BufferAttribute(this.vis,1));
    this.uniforms={uColor:{value:this.color},uTime:shared.uTime,uSize:{value:.019},uScale:shared.uScale,uOpacity:{value:1}};
    this.points=new THREE.Points(geo,new THREE.ShaderMaterial({vertexShader:POINT_VS,fragmentShader:POINT_FS,uniforms:this.uniforms,transparent:true,depthWrite:false,blending:THREE.AdditiveBlending}));this.points.frustumCulled=false;
    const skinGeo=new THREE.BufferGeometry();skinGeo.setAttribute('position',new THREE.BufferAttribute(this.meshPositions,3));skinGeo.setAttribute('normal',new THREE.BufferAttribute(this.meshNormals,3));skinGeo.setAttribute('vis',new THREE.BufferAttribute(this.meshVis,1));skinGeo.setIndex(indices);
    this.skin=new THREE.Mesh(skinGeo,new THREE.ShaderMaterial({vertexShader:SKIN_VS,fragmentShader:SKIN_FS,uniforms:this.uniforms,transparent:true,depthWrite:false,blending:THREE.AdditiveBlending,side:THREE.DoubleSide}));this.skin.frustumCulled=false;
    this.disc=new THREE.Mesh(new THREE.CircleGeometry(.72,48),new THREE.ShaderMaterial({vertexShader:DISC_VS,fragmentShader:DISC_FS,uniforms:{uColor:{value:this.color},uTime:shared.uTime,uOpacity:this.uniforms.uOpacity},transparent:true,depthWrite:false,blending:THREE.AdditiveBlending,side:THREE.DoubleSide}));this.disc.rotation.x=-Math.PI/2;
    this.group=new THREE.Group();this.group.add(this.skin,this.points,this.disc);
  }
  /**
   * pose: { x, z, yaw, phase, speed, strideX, strideZ, crouch, guard, screen, shoot, catchBall, jump, dribble, ballLocal }
   * 权重类字段取 0..1；ballLocal 为运球手目标（身体局部坐标）。jointsOnly=true 时只解骨架不重算点云。
   */
  setPose(p, jointsOnly = false, skipContacts = false) {
    if(this._setPresentationPose(p,jointsOnly,skipContacts?null:(p.ballContacts??p.ballContact)))return;
    this.gesture = p;
    this.rigHints=null;this.jointTrust={};this.chainDiagnostics={};this.contactDiagnostics=[];this._contactBySide={};
    const fwd = new THREE.Vector3(Math.sin(p.yaw), 0, Math.cos(p.yaw));
    const right = new THREE.Vector3(Math.cos(p.yaw), 0, -Math.sin(p.yaw));
    const base = new THREE.Vector3(p.x, 0, p.z);
    const W = (lx, ly, lz) => base.clone().addScaledVector(right, lx).addScaledVector(UP, ly).addScaledVector(fwd, lz);
    const J = this.joints;
    const c = p.crouch;
    const bob = Math.abs(Math.sin(p.phase)) * 0.04 * p.speed;
    const pelvisY = 1.0 - 0.26 * c + p.jump - bob;
    const lean = 0.12 * c + 0.06 * p.speed;

    J.pelvisLo = W(0, pelvisY - 0.1, 0.01 * c);
    J.pelvis = W(0, pelvisY, 0.02 * c);
    J.chest = W(0, pelvisY + 0.46 - 0.04 * c, lean);
    J.spine = W(0, pelvisY + 0.22 - 0.02 * c, lean * .42);
    J.neck = W(0, pelvisY + 0.6 - 0.05 * c, lean * 1.15);
    J.headBase = W(0, pelvisY + 0.66 - 0.05 * c, lean * 1.25);
    const chestY = pelvisY + 0.42 - 0.04 * c;

    for (const [side, s] of [['L', -1], ['R', 1]]) {
      const ph = p.phase + (s < 0 ? Math.PI : 0);
      // 腿：脚目标 → IK，膝盖朝前（下蹲时略外展）
      const hip = W(s * 0.1, pelvisY - 0.06, 0);
      const footY = 0.1 + Math.max(0, Math.cos(ph)) * 0.16 * p.speed + p.jump * 0.85;
      // 步幅沿身体局部的移动方向（防守横滑时脚也横着迈）
      const stride = Math.sin(ph) * 0.42 * p.speed;
      const foot = W(s * (0.12 + 0.2 * c) + (p.strideX ?? 0) * stride, footY, (p.strideZ ?? 1) * stride + 0.05 * c);
      const knee = solveIK(hip, foot, 0.47, 0.47, fwd.clone().addScaledVector(right, s * 0.3 * c), new THREE.Vector3());
      J[`hip${side}`] = hip;
      J[`knee${side}`] = knee;
      J[`ankle${side}`] = foot;
      // 脚尖朝身体前方并略外八
      J[`toeDir${side}`] = fwd.clone().addScaledVector(right, s * 0.18).normalize();

      // 手臂：多个动作目标按权重依次混合
      const sh = W(s * 0.2, chestY + 0.02, lean);
      const target = new THREE.Vector3(s * 0.25, chestY - 0.55, -Math.sin(ph) * 0.28 * p.speed + 0.08);
      const blend = (w, x, y, z) => { if (w > 0) target.lerp(new THREE.Vector3(x, y, z), w); };
      blend(p.guard, s * 0.6, chestY - 0.16, 0.34);        // 防守：屈肘张手，手在腰胸之间
      blend(p.screen, -s * 0.06, chestY - 0.24, 0.2);       // 掩护：双臂护在胸腹前
      blend(p.catchBall, s * 0.13, chestY - 0.02, 0.36);
      blend(p.shoot, s * 0.09, pelvisY + 1.32, 0.22);
      if (s > 0 && p.dribble > 0 && p.ballLocal) blend(p.dribble, ...p.ballLocal);
      const hand = W(target.x, target.y, target.z);
      const pole = new THREE.Vector3().addScaledVector(right, s * 0.7).addScaledVector(UP, -0.5).addScaledVector(fwd, -0.5);
      J[`shoulder${side}`] = sh;
      J[`elbow${side}`] = solveIK(sh, hand, 0.31, 0.28, pole, new THREE.Vector3());
      J[`hand${side}`] = hand;
    }

    if (this.bodyScale !== 1) {
      for (const [name, value] of Object.entries(J)) {
        if (name.startsWith('toeDir')) continue;
        value.sub(base).multiplyScalar(this.bodyScale).add(base);
      }
    }
    this.disc.position.set(p.x, 0.012, p.z);
    this.fwd = fwd;
    this.right = right;
    if(!skipContacts)this._applyContacts(p.ballContacts??p.ballContact);
    if (!jointsOnly) this._fill(this.fwd, this.right);
  }

  /** Retarget source-video temporal pose estimates, with confidence fallback.
   * They are model reconstructions, not an optical-tracking ground-truth rig.
   * Tiny fingers/face are not claimed to have been recovered from the video.
   */
  setTrackedPose(p,sample,jointsOnly=false,contact=null){
    if(this._setPresentationPose(p,jointsOnly,contact??p.ballContacts??p.ballContact))return;
    if(!sample?.jointsWorld?.length){this.setPose(p,jointsOnly);return;}
    const lookup=Object.fromEntries(sample.jointNames.map((name,i)=>[name,i]));
    const get=(name)=>{const i=lookup[name],xyz=sample.jointsWorld[i];if(i===undefined||!xyz||!xyz.every(Number.isFinite))return null;return{v:new THREE.Vector3(...xyz),c:sample.rig?.presentationConfidence?.[i]??(Number.isFinite(sample.confidence?.[i])?sample.confidence[i]:0)};};
    const coverageValue=sample.coverageWeight??1;const rawCoverage=Number.isFinite(coverageValue)?clamp(coverageValue,0,1):0;
    const coverage=smooth(rawCoverage),left=get('leftShoulder'),right=get('rightShoulder');
    const heading=sample.rig?.headingYaw??(sample.forwardWorld?Math.atan2(sample.forwardWorld[0],sample.forwardWorld[2]):p.yaw);
    const headingSupport=coverage*smooth(((sample.rig?.headingConfidence??sample.orientationConfidence??0)-.16)/.40);
    const aligned={...p,yaw:p.yaw+relativeAngle(heading,p.yaw)*headingSupport};
    this.setPose(aligned,true,true);this.gesture=p;this.rigHints=sample.rig??null;
    const J=this.joints,mapping={pelvis:'pelvis',spine:'spine',chest:'thorax',neck:'neck',headBase:'head',hipL:'leftHip',hipR:'rightHip',kneeL:'leftKnee',kneeR:'rightKnee',ankleL:'leftAnkle',ankleR:'rightAnkle',shoulderL:'leftShoulder',shoulderR:'rightShoulder',elbowL:'leftElbow',elbowR:'rightElbow',handL:'leftWrist',handR:'rightWrist'};
    const sources={},weight={},sourcePelvis=get('pelvis');let total=0;
    for(const [rig,name]of Object.entries(mapping)){const source=get(name);if(!source)continue;sources[rig]=source;weight[rig]=coverage*smooth((source.c-.16)/.45);this.jointTrust[rig]=source.c>=.60&&rawCoverage>=.999999;if(this.jointTrust[rig])weight[rig]=1;total+=weight[rig];}
    if(total===0){this.setPose(p,true,true);this.rigHints=sample.rig??null;this._applyContacts(contact??p.ballContacts??p.ballContact);if(!jointsOnly)this._fill(this.fwd,this.right);return;}
    // Anchor the complete fallback to one pelvis before mixing local offsets.
    // A hidden ankle/hand follows its attached body, never a second world pose.
    const rootShift=sourcePelvis?sourcePelvis.v.clone().sub(J.pelvis).multiplyScalar(weight.pelvis??0):new THREE.Vector3();
    for(const [name,v]of Object.entries(J))if(!name.startsWith('toeDir'))v.add(rootShift);
    const fallback=Object.fromEntries(Object.entries(J).map(([k,v])=>[k,v.clone()]));
    if(sources.chest){const shift=sources.chest.v.clone().sub(J.chest).multiplyScalar(weight.chest??0);for(const side of ['L','R'])for(const name of ['shoulder','elbow','hand'])J[name+side].add(shift);}
    for(const [rig,source]of Object.entries(sources)){if(rig==='pelvis')continue;J[rig].lerp(source.v,weight[rig]);}
    const headAxis=unit(J.headBase.clone().sub(J.neck));J.headBase.addScaledVector(headAxis,-.055*this.bodyScale*(weight.headBase??0));
    const spineAxis=unit(J.spine.clone().sub(J.pelvis));J.pelvisLo.copy(J.pelvis).addScaledVector(spineAxis,-.1*this.bodyScale);
    this._retargetChains(fallback,sources,rawCoverage);
    for(const side of ['L','R']){const shin=J['ankle'+side].clone().sub(J['knee'+side]);shin.y=0;J['toeDir'+side]=unit(this.fwd.clone().addScaledVector(shin,-.45),this.fwd);}
    this._applyContacts(contact??p.ballContacts??p.ballContact);
    if(!jointsOnly)this._fill(this.fwd,this.right);
  }

  _retargetChains(fallback,sources={},coverage=0){
    const J=this.joints,s=this.bodyScale;
    for(const side of ['L','R'])for(const kind of ['arm','leg']){
      const arm=kind==='arm',names=[(arm?'shoulder':'hip')+side,(arm?'elbow':'knee')+side,(arm?'hand':'ankle')+side],points=names.map(n=>J[n]),pins=names.map(n=>this.jointTrust[n]===true);
      const before=points.map(p=>p.clone()),priors=this.rigHints?.limbLengths?.[kind+side]??(arm?[.31*s,.28*s]:[.47*s,.47*s]);
      const [a,m,end]=points;
      // Confidence is a continuous weight, never a switch between incompatible
      // bone projections. Reliable endpoints retain their observed lengths;
      // uncertain chains approach their anatomical prior smoothly.
      const lengths=priors.map((prior,index)=>{
        const first=sources[names[index]],second=sources[names[index+1]];
        const trust=smooth((Math.min(first?.c??0,second?.c??0)-.50)/.25)*smooth(coverage);
        const observed=first&&second?first.v.distanceTo(second.v):prior;
        const plausible=clamp(observed,prior*.72,prior*1.28);
        return THREE.MathUtils.lerp(prior,plausible,trust);
      });
      const [l1,l2]=lengths;
      const axis=unit(end.clone().sub(a)),pole=m.clone().sub(a);
      const bend=pole.clone().addScaledVector(axis,-pole.dot(axis));
      const sign=side==='L'?-1:1;
      const fallbackPole=arm?this.right.clone().multiplyScalar(sign*.7).addScaledVector(UP,-.5).addScaledVector(this.fwd,-.5):this.fwd.clone().addScaledVector(this.right,sign*.12);
      // Near extension the tiny bend direction is ill-conditioned. Fade to a
      // transported source-time pole rather than letting it choose a new side.
      const guide=this.rigHints?.limbFrames?.[kind+side];
      let stable=guide?.bendPole?vectorOf(guide.bendPole):fallbackPole;
      stable=transverse(axis,stable,fallbackPole);
      const actual=transverse(axis,bend,stable);
      const confidence=Math.min(...names.map(name=>sources[name]?.c??0));
      const support=smooth((confidence-.55)/.25)*smooth(coverage)*smooth((bend.length()-.008*s)/(.05*s))*smooth((actual.dot(stable)+.2)/.6);
      pole.copy(stable).lerp(actual,support).normalize();
      solveIK(a,end,l1,l2,pole,m);
      this.chainDiagnostics[kind+side]={pins,priorLengths:priors.slice(),lengths:[a.distanceTo(m),m.distanceTo(end)],correctionsMeters:points.map((p,i)=>p.distanceTo(before[i])),relaxedPinnedEndpoints:false};
    }
  }

  _setPresentationPose(p,jointsOnly,contacts){
    const cache=this.presentationFrames;
    if(!cache||!Number.isFinite(p.time))return false;
    const time=clamp(p.time,0,cache.duration),position=time*cache.rate;
    const first=Math.min(cache.count,Math.floor(position)),second=Math.min(cache.count,first+1);
    const span=Math.min(cache.duration,second/cache.rate)-first/cache.rate;
    const amount=span>0?clamp((time-first/cache.rate)/span,0,1):0;
    const a=cache.frames[first],b=cache.frames[second],J=this.joints;
    this.gesture=p;this.rigHints=null;this.jointTrust={};this.chainDiagnostics={};this._contactBySide={};this.contactDiagnostics=[];
    for(const name of cache.anchors)J[name]=a.joints[name].clone().lerp(b.joints[name],amount);
    const yaw=THREE.MathUtils.lerp(a.yaw,b.yaw,amount);
    this.fwd=new THREE.Vector3(Math.sin(yaw),0,Math.cos(yaw));this.right=new THREE.Vector3(Math.cos(yaw),0,-Math.sin(yaw));
    for(const [name,names]of Object.entries(cache.chains)){
      const lengths=[];
      for(let bone=0;bone<2;bone++){
        const qa=a.chains[name][bone],qb=b.chains[name][bone],q=new THREE.Quaternion().slerpQuaternions(qa.rotation,qb.rotation,amount);
        const length=THREE.MathUtils.lerp(qa.length,qb.length,amount);lengths.push(length);
        J[names[bone+1]]=J[names[bone]].clone().addScaledVector(new THREE.Vector3(0,1,0).applyQuaternion(q),length);
      }
      this.chainDiagnostics[name]={lengths,presentation:'source-time-bone-direction-filter',windowSigmaSeconds:cache.sigma};
    }
    for(const side of ['L','R']){
      const shin=J['ankle'+side].clone().sub(J['knee'+side]);shin.y=0;
      J['toeDir'+side]=unit(this.fwd.clone().addScaledVector(shin,-.45),this.fwd);
    }
    this.disc.position.set(p.x,.012,p.z);
    const contactList=contacts?(Array.isArray(contacts)?contacts:[contacts]):[];
    const active=new Map(contactList.map(contact=>[contact.side+':'+contact.mode,contact]));
    const fields=['target','palmTarget','palmNormal','palmAlong','ball'];
    const hintAt=key=>{
      const first=a.contactHints[key]??b.contactHints[key],second=b.contactHints[key]??first;if(!first)return null;
      const hint={...first,time:p.time};for(const field of fields){const x=vectorOf(first[field]),y=vectorOf(second[field]);if(x&&y)hint[field]=x.lerp(y,amount);}
      return hint;
    };
    // Preserve the finite release tail when reach gating omits a payload.
    // Interpolate every target channel on the source clock; using a live
    // target only while its input payload exists would reintroduce a pop.
    if(contacts!==undefined&&contacts!==null)for(const key of Object.keys(a.contactWeights)){
      const hint=hintAt(key);if(!hint)continue;
      // All prepared target channels share the same continuous interpolation,
      // including full-weight catch/gather switches in the input event list.
      // The source ball itself remains sampled directly by BallSolver.
      active.set(key,hint);
    }
    const bySide=new Map();for(const [key,contact]of active){
      const weight=key in a.contactWeights||key in b.contactWeights?THREE.MathUtils.lerp(a.contactWeights[key]??0,b.contactWeights[key]??0,amount):contact.weight;
      if(weight>0){if(!bySide.has(contact.side))bySide.set(contact.side,[]);bySide.get(contact.side).push({...contact,weight});}
    }
    const resolved=[];for(const list of bySide.values()){
      // Catch/gather and release/dribble may overlap. Blend their targets
      // rather than switching an entire palm orientation at an argmax tie.
      const strongest=list.reduce((a,b)=>a.weight>=b.weight?a:b),combined={...strongest};
      for(const field of fields){const vector=new THREE.Vector3();let total=0;
        for(const contact of list){const value=vectorOf(contact[field]);if(value){const w=contact.weight**4;vector.addScaledVector(value,w);total+=w;}}
        if(total>0)combined[field]=vector.divideScalar(total);
      }resolved.push(combined);
    }
    this._applyContacts(resolved);
    if(!jointsOnly)this._fill(this.fwd,this.right);
    return true;
  }

  prepareAppearanceFrames(poseAt,motionAt,duration,rate=120){
    // The presentation rig is derived once from the confidence-constrained
    // reconstruction. Filtering bone rotations (not mesh points) preserves
    // anatomical lengths and full poses while spreading occlusion switches
    // over a short, symmetric source-time window. Raw observations stay intact.
    this.appearanceFrames=null;this.presentationFrames=null;
    const count=Math.ceil(duration*rate),sigma=.05,radius=Math.ceil(3*sigma*rate);
    const chains=Object.fromEntries(['armL','armR','legL','legR'].map(name=>{
      const arm=name.startsWith('arm'),side=name.at(-1);
      return[name,[(arm?'shoulder':'hip')+side,(arm?'elbow':'knee')+side,(arm?'hand':'ankle')+side]];
    }));
    const anchors=['pelvisLo','pelvis','spine','chest','neck','headBase','hipL','hipR','shoulderL','shoulderR'];
    const raw=[],previous={},contactKeys=new Set();let previousYaw;
    for(let index=0;index<=count;index++){
      const time=Math.min(duration,index/rate),input=poseAt(time),pose={...input,time,ballContact:null,ballContacts:null},motion=motionAt(time);
      if(motion)this.setTrackedPose(pose,motion,true);else this.setPose(pose,true);
      let yaw=Math.atan2(this.fwd.x,this.fwd.z);if(previousYaw!==undefined)yaw=previousYaw+relativeAngle(yaw,previousYaw);previousYaw=yaw;
      const contactWeights={},contacts={};for(const contact of input.ballContacts??[]){const key=contact.side+':'+contact.mode;contactWeights[key]=contact.weight??0;contacts[key]=contact;contactKeys.add(key);}
      const row={joints:Object.fromEntries(anchors.map(name=>[name,this.joints[name].clone()])),yaw,chains:{},contactWeights,contacts};
      for(const [name,names]of Object.entries(chains))row.chains[name]=[0,1].map(bone=>{
        const vector=this.joints[names[bone+1]].clone().sub(this.joints[names[bone]]),length=vector.length(),axis=unit(vector);
        const key=name+bone,prior=previous[key],lateral=prior?transport(prior.lateral,prior.axis,axis):transverse(axis,this.right,this.fwd);
        const rotation=new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(lateral,axis,new THREE.Vector3().crossVectors(lateral,axis)));
        if(prior&&rotation.dot(prior.rotation)<0)rotation.set(-rotation.x,-rotation.y,-rotation.z,-rotation.w);
        previous[key]={axis,lateral,rotation};return{length,rotation};
      });raw.push(row);
    }
    const frames=raw.map((row,index)=>{
      const weights=[];let total=0;
      for(let j=Math.max(0,index-radius);j<=Math.min(count,index+radius);j++){
        const weight=Math.exp(-.5*((j-index)/(rate*sigma))**2);weights.push([j,weight]);total+=weight;
      }
      const result={joints:{},chains:{},contactWeights:{},contactHints:{},yaw:weights.reduce((sum,[j,w])=>sum+raw[j].yaw*w,0)/total};
      // Contact targets remain on the ball at the current time. Only the
      // confidence/reach gate gets a brief symmetric fade before final IK.
      for(const key of contactKeys){let value=0,sum=0,nearest=Infinity;for(let j=Math.max(0,index-7);j<=Math.min(count,index+7);j++){const w=Math.exp(-.5*((j-index)/(rate*.018))**2);value+=(raw[j].contactWeights[key]??0)*w;sum+=w;if(raw[j].contacts[key]&&Math.abs(j-index)<nearest){result.contactHints[key]=raw[j].contacts[key];nearest=Math.abs(j-index);}}result.contactWeights[key]=value/sum;}
      for(const name of anchors)result.joints[name]=weights.reduce((sum,[j,w])=>sum.addScaledVector(raw[j].joints[name],w/total),new THREE.Vector3());
      for(const name of Object.keys(chains))result.chains[name]=[0,1].map(bone=>{
        const values=[0,0,0,0];let length=0;
        for(const [j,w]of weights){const source=raw[j].chains[name][bone];length+=source.length*w/total;source.rotation.toArray().forEach((v,k)=>values[k]+=v*w);}
        return{length,rotation:new THREE.Quaternion(...values).normalize()};
      });return result;
    });
    this.presentationFrames={rate,duration,count,sigma,chains,anchors,frames};
    // The envelope's transported cross-sections use that same presentation
    // skeleton, after bounded palm contact. Absolute-time sampling remains
    // independent of playback order, including backward and random seeking.
    const gauges=Object.fromEntries(Object.keys(chains).map(name=>[name,[]])),twists=Object.fromEntries(Object.keys(chains).map(name=>[name,[]])),prior={};
    for(let index=0;index<=count;index++){
      const time=Math.min(duration,index/rate),pose={...poseAt(time),time};this.setPose(pose,true);
      for(const [name,names]of Object.entries(chains)){
        const axis=unit(this.joints[names[1]].clone().sub(this.joints[names[0]])),previous=prior[name];
        const lateral=previous?transport(previous.lateral,previous.axis,axis):transverse(axis,this.right,this.fwd);
        gauges[name].push(new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(lateral,axis,new THREE.Vector3().crossVectors(lateral,axis))));
        const lowerAxis=unit(this.joints[names[2]].clone().sub(this.joints[names[1]]));
        const fromUpper=transport(lateral,axis,lowerAxis);
        const lowerLateral=previous?transport(previous.lowerLateral,previous.lowerAxis,lowerAxis):fromUpper;
        let twist=Math.atan2(new THREE.Vector3().crossVectors(fromUpper,lowerLateral).dot(lowerAxis),fromUpper.dot(lowerLateral));
        if(previous)twist=previous.twist+relativeAngle(twist,previous.twist);
        // Let harmless axial gauge drift relax gradually; otherwise a long
        // sequence of bent poses could wind an entire turn into the elbow.
        const relaxed=twist*Math.exp(-1/(rate*.12));
        lowerLateral.applyAxisAngle(lowerAxis,relaxed-twist);twist=relaxed;
        twists[name].push(twist);prior[name]={axis,lateral,lowerAxis,lowerLateral,twist};
      }
    }
    this.appearanceFrames={rate,duration,chains:gauges,sample(time,name){
      if(!Number.isFinite(time)||!gauges[name])return null;
      const position=clamp(time,0,duration)*rate,first=Math.min(count,Math.floor(position)),second=Math.min(count,first+1);
      const actualSpan=Math.min(duration,second/rate)-first/rate;
      const amount=actualSpan>0?clamp((time-first/rate)/actualSpan,0,1):0;
      const quaternion=new THREE.Quaternion().slerpQuaternions(gauges[name][first],gauges[name][second],amount);
      return{axis:new THREE.Vector3(0,1,0).applyQuaternion(quaternion),lateral:new THREE.Vector3(1,0,0).applyQuaternion(quaternion),twist:THREE.MathUtils.lerp(twists[name][first],twists[name][second],amount)};
    }};
    return this.appearanceFrames;
  }

  _limbGuide(kind,side){
    const J=this.joints,arm=kind==='arm',a=J[(arm?'shoulder':'hip')+side],m=J[(arm?'elbow':'knee')+side],axis=unit(m.clone().sub(a));
    const guide=this.appearanceFrames?.sample(this.gesture?.time,kind+side)??this.rigHints?.limbFrames?.[kind+side];let lateral;
    if(guide){const sourceAxis=vectorOf(guide.axis),sourceLateral=vectorOf(guide.lateral);lateral=transport(sourceLateral,sourceAxis,axis);}
    else{const end=J[(arm?'hand':'ankle')+side],targetAxis=unit(end.clone().sub(a));
      const sign=side==='L'?-1:1,pole=arm?this.right.clone().multiplyScalar(sign*.7).addScaledVector(UP,-.5).addScaledVector(this.fwd,-.5):this.fwd.clone().addScaledVector(this.right,sign*.12);
      const bend=transverse(targetAxis,pole,this.fwd);lateral=unit(new THREE.Vector3().crossVectors(targetAxis,bend),this.right);lateral=transport(lateral,targetAxis,axis);
    }
    return{axis,lateral,twist:guide?.twist??0};
  }

  getHandFrame(side='R',contactHint=null){
    const J=this.joints,wrist=J['hand'+side];if(!wrist)return null;
    const lower=unit(wrist.clone().sub(J['elbow'+side])),guide=this._limbGuide('arm',side);
    let along=lower.clone(),across=transport(guide.lateral,guide.axis,along).applyAxisAngle(along,guide.twist);
    const contact=contactHint??this._contactBySide?.[side],normalHint=vectorOf(contact?.palmNormal),alongHint=vectorOf(contact?.palmAlong),weight=clamp(contact?.weight??0,0,1);
    if(normalHint){unit(normalHint);let desired=alongHint??along.clone().addScaledVector(normalHint,-along.dot(normalHint));if(desired.lengthSq()<1e-6)desired=this.fwd.clone().addScaledVector(normalHint,-this.fwd.dot(normalHint));unit(desired,this.right);
      const angle=along.angleTo(desired),limit=THREE.MathUtils.degToRad(95),fraction=angle>1e-6?Math.min(1,limit/angle)*weight:0;
      along.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(along,desired).slerp(new THREE.Quaternion(),1-fraction));
      across=transport(across,lower,along);
      const desiredAcross=unit(new THREE.Vector3().crossVectors(normalHint,along),across);
      const angleAcross=Math.atan2(new THREE.Vector3().crossVectors(across,desiredAcross).dot(along),across.dot(desiredAcross));
      across=across.applyAxisAngle(along,clamp(angleAcross,-Math.PI*.75,Math.PI*.75)*weight);
    }
    const s=this.bodyScale,lowerY=-clamp((wrist.y-.055*s)/(.20*s),0,1),flex=contactHeight(along.y,lowerY,.065)[0];
    if(flex>along.y+1e-7){const horizontal=unit(new THREE.Vector3(along.x,0,along.z),this.fwd),target=horizontal.multiplyScalar(Math.sqrt(Math.max(0,1-flex*flex))).addScaledVector(UP,flex),old=along.clone();along.lerp(target,1-smooth((wrist.y-.17*s)/(.085*s))).normalize();across=transport(across,old,along);}
    across=transverse(along,across,this.right);const depth=unit(new THREE.Vector3().crossVectors(across,along)),normal=depth.clone().negate();
    return{wrist:wrist.clone(),palmCenter:wrist.clone().addScaledVector(along,.052*s),along,across,depth,normal,bodyScale:s};
  }
  getPalmPosition(side='R',contactHint=null){return this.getHandFrame(side,contactHint)?.palmCenter??null;}

  _applyContacts(contacts){
    this.contactDiagnostics=[];if(!contacts)return;const list=Array.isArray(contacts)?contacts:[contacts];
    for(const contact of list){const side=contact.side==='L'?'L':'R',weight=smooth(clamp(contact.weight??0,0,1));if(weight===0)continue;
      const J=this.joints,a=J['shoulder'+side],m=J['elbow'+side],end=J['hand'+side],before=end.clone(),l1=a.distanceTo(m),l2=m.distanceTo(end);
      this._contactBySide[side]={...contact,weight};let target=vectorOf(contact.target),palmTarget=vectorOf(contact.palmTarget);
      if(palmTarget){const frame=this.getHandFrame(side);target=palmTarget.clone().sub(frame.palmCenter.clone().sub(frame.wrist));}
      if(!target)continue;const maxCorrection=clamp(contact.maxCorrectionMeters??.18,0,.24),delta=target.clone().sub(before).multiplyScalar(weight);if(delta.length()>maxCorrection)delta.setLength(maxCorrection);
      const desired=before.clone().add(delta),pole=m.clone().sub(a);solveIK(a,desired,l1,l2,pole,m);end.copy(desired);
      if(end.distanceTo(before)>maxCorrection+1e-9){const limited=before.clone().add(end.clone().sub(before).setLength(maxCorrection));solveIK(a,limited,l1,l2,pole,m);end.copy(limited);}
      this.contactDiagnostics.push({side,mode:contact.mode,time:contact.time,appliedCorrectionMeters:end.distanceTo(before),maxCorrectionMeters:maxCorrection,residualMeters:palmTarget?this.getPalmPosition(side).distanceTo(palmTarget):end.distanceTo(target),upperLength:l1,lowerLength:l2});
    }
  }

  _buildEnvelopes(fwd,right) {
    const J=this.joints,s=this.bodyScale,g=this.gesture??{};
    const hipRight=new THREE.Vector3().subVectors(J.hipR,J.hipL).normalize();
    const shoulderRight=new THREE.Vector3().subVectors(J.shoulderR,J.shoulderL).normalize();

    const spineUp=new THREE.Vector3().subVectors(J.neck,J.pelvisLo).normalize();
    const headUp=new THREE.Vector3().subVectors(J.headBase,J.neck).normalize();if(headUp.lengthSq()<.5)headUp.copy(spineUp);
    const headRight=shoulderRight.clone().addScaledVector(headUp,-shoulderRight.dot(headUp)).normalize();
    const headForward=new THREE.Vector3().crossVectors(headRight,headUp).normalize();
    const shoulderWidth=clamp(J.shoulderR.distanceTo(J.shoulderL)/(.4*s),.88,1.18);
    const torsoKnots=[[0,J.pelvisLo.clone().addScaledVector(spineUp,-.025*s)],[.15,J.pelvis],[.45,J.spine],[.78,J.chest],[1,J.neck]];
    const palmFrames=Object.fromEntries(['L','R'].map(side=>{const frame=this.getHandFrame(side);return[side,{...frame,base:frame.wrist}];}));
    for(const part of this.parts){
      let previousTangent=null,previousLateral=null;
      const side=part.side,hand=palmFrames[side],sign=side==='L'?-1:1;
      const stableGuide=/^(arm|leg)/.test(part.name)?this._limbGuide(part.name.startsWith('arm')?'arm':'leg',side):null;
      for(let i=0;i<=part.rings;i++){
        const u=i/part.rings;let c,tangent,lateral=right,rx,rz;
        if(part.name==='torso'){
          ({c,tangent}=splineAt(torsoKnots,u));lateral=hipRight.clone().lerp(shoulderRight,smooth((u-.2)/.6)).normalize();
          rx=profileAt([.06,.164,.175,.160,.145,.157,.184,.209,.220,.135,.056],u)*s;
          rx*=1+(shoulderWidth-1)*smooth((u-.45)/.32);
          rz=profileAt([.07,.108,.116,.103,.095,.107,.124,.128,.107,.083,.054],u)*s;
        }else if(part.name==='neck'){
          c=J.neck.clone().lerp(J.headBase.clone().addScaledVector(headUp,.04*s),u);tangent=headUp;lateral=headRight;
          rx=profileAt([.055,.050,.048,.055],u)*s;rz=profileAt([.055,.051,.048,.050],u)*s;
        }else if(part.name==='head'){
          c=J.headBase.clone().addScaledVector(headUp,(-.028+u*.239)*s).addScaledVector(headForward,.011*s);
          tangent=headUp;lateral=headRight;
          rx=profileAt([.018,.059,.072,.079,.087,.090,.087,.076,.051,.003],u)*s;
          rz=profileAt([.026,.061,.078,.089,.096,.094,.088,.076,.047,.003],u)*s;
        }else if(part.name.startsWith('arm')||part.name.startsWith('leg')){
          const arm=part.name.startsWith('arm');
          const anchor=J[(arm?'shoulder':'hip')+side],b=J[(arm?'elbow':'knee')+side],end=J[(arm?'hand':'ankle')+side];
          const boneAxis=new THREE.Vector3().subVectors(b,anchor).normalize();
          const extension=(arm?.054:.075)*s;
          const a=anchor.clone().addScaledVector(boneAxis,-extension);
          const h=hingeAt(a,b,end,u,(arm?.048:.058)*s);({c,tangent}=h);
          part.hinge=h.hinge;part.root=a;part.hingeJoint=b;
          if(arm){
            rx=profileAt(h.upper?[.071,.076,.073,.063,.048,.043]:[.043,.052,.052,.043,.034,.027],h.local)*s;
            rz=profileAt(h.upper?[.073,.073,.068,.058,.043,.041]:[.041,.052,.047,.039,.030,.026],h.local)*s;
            if(h.upper){const muscle=Math.sin(Math.PI*h.local)**2*Math.min(1,h.bend/1.9);rx+=.006*muscle*s;rz+=.009*muscle*s;}
          }else{
            rx=profileAt(h.upper?[.102,.113,.110,.101,.080,.063]:[.063,.075,.074,.060,.048,.037],h.local)*s;
            rz=profileAt(h.upper?[.113,.116,.108,.091,.071,.064]:[.064,.083,.082,.065,.049,.041],h.local)*s;
            // The patella is part of the shin/thigh envelope, not a separate ball.
            c.addScaledVector(fwd,.009*s*Math.exp(-Math.pow((u-h.hinge)/.06,2)));
          }
          // A proximal cap is skinned into the chest/pelvis, making the deltoid
          // and hip transitions rounded rather than an exposed tube opening.
          const capLength=extension/(a.distanceTo(b)||1);
          if(h.upper&&h.local<capLength){const q=h.local/capLength;const cap=Math.sqrt(Math.max(.0001,1-(1-q)**2));rx*=cap;rz*=cap;}
        }else if(part.name.startsWith('palm')){
          c=hand.base.clone().addScaledVector(hand.along,.098*s*u);tangent=hand.along;lateral=hand.across;
          rx=profileAt([.027,.031,.041,.039,.032],u)*s;rz=profileAt([.025,.023,.021,.018,.014],u)*s;
        }else if(part.name.startsWith('finger')){
          const finger=part.finger,thumb=finger===4;
          const curl=clamp(.10+.30*(g.catchBall??0)+.24*(g.screen??0)+.12*(g.dribble??0)-.07*(g.guard??0),.03,.60);
          const length=(thumb?.071:[.066,.081,.075,.057][finger])*s;
          const offset=(thumb?-sign*.033:[-.027,-.009,.010,.027][finger])*s;
          const root=hand.base.clone().addScaledVector(hand.along,(thumb?.044:.088)*s).addScaledVector(hand.across,offset);
          const extension=hand.along.clone().multiplyScalar(thumb?.62:1).addScaledVector(hand.across,thumb?-sign*.72:([-.11,-.035,.04,.12][finger]*(1-(g.catchBall??0)))).normalize();
          const bend=hand.depth.clone().multiplyScalar(curl*length);
          c=root.clone().addScaledVector(extension,length*u).addScaledVector(bend,u*u);
          tangent=extension.clone().addScaledVector(bend,2*u/length).normalize();lateral=hand.across;
          const contact=contactHeight(c.y,.024*s,.012*s);c.y=contact[0];tangent.y*=contact[1];tangent.normalize();
          rx=(thumb?.0090:.0073)*s*profileAt([.90,1,.95,.83,.60,.08],u);rz=rx*.87;
        }else if(part.name.startsWith('shoe')){
          const ankle=J['ankle'+side],toe=J['toeDir'+side].clone();
          // A planted sole is flat; airborne feet can plantar-flex slightly.
          toe.y=-.16*smooth((ankle.y-.15*s)/(.28*s));toe.normalize();
          c=ankle.clone().addScaledVector(toe,(-.075+.286*u)*s).addScaledVector(UP,-.037*s);
          tangent=toe;lateral=new THREE.Vector3().crossVectors(UP,toe).normalize();
          rx=profileAt([.009,.038,.048,.058,.061,.058,.048,.020,.004],u)*s;
          rz=profileAt([.012,.040,.063,.058,.044,.040,.033,.020,.004],u)*s;
          part.soleY=Math.max(.012*s,ankle.y-.085*s);
        }
        if(part.name.startsWith('arm')||part.name.startsWith('leg')){
          lateral=previousTangent?transport(previousLateral,previousTangent,tangent):transport(stableGuide.lateral,stableGuide.axis,tangent);
        }else if(part.name==='torso'&&previousTangent)lateral=transport(previousLateral,previousTangent,tangent);
        previousTangent=tangent.clone();previousLateral=lateral.clone();
        if(stableGuide?.twist)lateral.applyAxisAngle(tangent,stableGuide.twist*smooth((u-.35)/.3));
        putRing(part.frames,i,c,tangent,lateral,rx,rz,u);
      }
      finishRings(part);
    }
  }

  _surface(pi,u,cos,sin,out,norm) {
    const part=this.parts[pi],d=part.frames,r=clamp(u,0,1)*part.rings;
    const i=Math.min(part.rings-1,Math.floor(r)),w=r-i,a=i*STRIDE,b=a+STRIDE;
    const v=(k)=>d[a+k]+(d[b+k]-d[a+k])*w;
    const rx=v(12),rz=v(13),x0=v(3),x1=v(4),x2=v(5),z0=v(6),z1=v(7),z2=v(8);
    let px=cos*rx,pz=sin*rz;
    if(part.name==='torso'){
      // Paired pectoral volumes, a shallow sternum and shoulder-blade planes.
      const chest=Math.exp(-Math.pow((u-.69)/.16,2)),side=Math.exp(-Math.pow((Math.abs(cos)-.48)/.34,2));
      pz+=(sin>0?1:-1)*chest*side*.011*this.bodyScale*Math.abs(sin);
      pz-=Math.max(0,sin)*chest*Math.exp(-Math.pow(cos/.22,2))*.005*this.bodyScale;
    }else if(part.name==='head'&&sin>0){
      // Jaw / cheek / brow / nose remain restrained at basketball-player scale.
      const front=sin**12;
      pz+=front*this.bodyScale*(.017*Math.exp(-Math.pow((u-.47)/.085,2))+.005*Math.exp(-Math.pow((u-.63)/.07,2)));
      px*=1-.05*Math.exp(-Math.pow((u-.23)/.09,2));
    }
    out[0]=v(0)+x0*px+z0*pz;out[1]=v(1)+x1*px+z1*pz;out[2]=v(2)+x2*px+z2*pz;
    const nx=cos/(rx||.001),nz=sin/(rz||.001),nt=-(cos*cos*v(14)/(rx||.001)+sin*sin*v(15)/(rz||.001));
    norm[0]=x0*nx+z0*nz+v(9)*nt;norm[1]=x1*nx+z1*nz+v(10)*nt;norm[2]=x2*nx+z2*nz+v(11)*nt;
    const nl=Math.hypot(...norm)||1;norm[0]/=nl;norm[1]/=nl;norm[2]/=nl;
    if(part.name.startsWith('palm')||part.name.startsWith('finger')){
      const contact=contactHeight(out[1],.013*this.bodyScale,.012*this.bodyScale);out[1]=contact[0];
      if(contact[1]<.0001){norm[0]=0;norm[1]=-1;norm[2]=0;}else if(contact[1]<1){
        norm[1]/=contact[1];const len=Math.hypot(...norm)||1;norm[0]/=len;norm[1]/=len;norm[2]/=len;
      }
    }
    if(part.name.startsWith('shoe')&&out[1]<part.soleY){out[1]=part.soleY;norm[0]=0;norm[1]=-1;norm[2]=0;}
  }

  _skinField(part,x,y,z,maxFraction=1) {
    const d=part.frames,last=Math.floor(part.rings*maxFraction);let best=Infinity,k=0;
    for(let i=0;i<=last;i++){const j=i*STRIDE,e=(x-d[j])**2+(y-d[j+1])**2+(z-d[j+2])**2;if(e<best){best=e;k=j;}}
    const qx=x-d[k],qy=y-d[k+1],qz=z-d[k+2],rx=Math.max(d[k+12],.003*this.bodyScale),rz=Math.max(d[k+13],.003*this.bodyScale);
    const vx=qx*d[k+3]+qy*d[k+4]+qz*d[k+5],vz=qx*d[k+6]+qy*d[k+7]+qz*d[k+8];
    const ax=vx/rx,az=vz/rz,rad=Math.hypot(ax,az),along=qx*d[k+9]+qy*d[k+10]+qz*d[k+11];
    if((k===0&&along<-.010*this.bodyScale)||(k===last*STRIDE&&along>.030*this.bodyScale))return null;
    const nx=vx/(rx*rx),nz=vz/(rz*rz),nt=-(ax*ax*d[k+14]/rx+az*az*d[k+15]/rz)/(rad*rad||1);
    let n0=d[k+3]*nx+d[k+6]*nz+d[k+9]*nt,n1=d[k+4]*nx+d[k+7]*nz+d[k+10]*nt,n2=d[k+5]*nx+d[k+8]*nz+d[k+11]*nt;
    const len=Math.hypot(n0,n1,n2)||1;n0/=len;n1/=len;n2/=len;
    return [(rad-1)*Math.min(rx,rz),n0,n1,n2];
  }

  _blendSeam(pi,u,point,normal) {
    const part=this.parts[pi],s=this.bodyScale;let field=null,blendWidth=.038*s;
    if((part.name.startsWith('arm')&&u<.22)||(part.name.startsWith('leg')&&u<.19)){
      field=this._skinField(this.parts[0],...point);blendWidth=(part.name.startsWith('arm')?.046:.038)*s;
    }else if(part.name==='torso'){
      for(const side of ['L','R'])for(const kind of ['arm','leg']){
        const neighbour=this.parts.find(p=>p.name===kind+side),d=neighbour.frames;
        if(Math.hypot(point[0]-d[0],point[1]-d[1],point[2]-d[2])>.235*s)continue;
        const f=this._skinField(neighbour,...point,.22);if(f&&(!field||f[0]<field[0]))field=f;
      }
    }
    if(!field||field[0]>=blendWidth)return 1;
    const distance=field[0];if(distance<-blendWidth*.55)return 0;
    // Blend both surface gradients at their intersection. The softly expanded
    // envelope removes bright sleeve/hip cut lines while preserving volume.
    const w=.5*(1-smooth(distance/blendWidth));
    normal[0]=normal[0]*(1-w)+field[1]*w;normal[1]=normal[1]*(1-w)+field[2]*w;normal[2]=normal[2]*(1-w)+field[3]*w;
    const len=Math.hypot(...normal)||1;normal[0]/=len;normal[1]/=len;normal[2]/=len;
    const expansion=blendWidth*.23*(1-smooth(Math.abs(distance)/blendWidth));
    point[0]+=normal[0]*expansion;point[1]+=normal[1]*expansion;point[2]+=normal[2]*expansion;
    return smooth((distance+blendWidth*.40)/(blendWidth*.65));
  }

  _fill(fwd,right) {
    this._buildEnvelopes(fwd,right);
    const point=[0,0,0],normal=[0,0,0];
    const write=(tpl,pos,nrm,vis)=>{
      for(let i=0;i<vis.length;i++){const q=i*4,pi=tpl[q]|0,u=tpl[q+1];this._surface(pi,u,tpl[q+2],tpl[q+3],point,normal);
        const visibility=this._blendSeam(pi,u,point,normal);
        const k=i*3;pos[k]=point[0];pos[k+1]=point[1];pos[k+2]=point[2];nrm[k]=normal[0];nrm[k+1]=normal[1];nrm[k+2]=normal[2];vis[i]=visibility;}
    };
    write(this.tplData,this.positions,this.normals,this.vis);write(this.meshTplData,this.meshPositions,this.meshNormals,this.meshVis);
    for(const attr of Object.values(this.points.geometry.attributes))attr.needsUpdate=true;
    for(const attr of Object.values(this.skin.geometry.attributes))attr.needsUpdate=true;
  }

  /** 身体局部坐标 → 世界坐标（用于球贴手）。 */
  toWorld(x, y, z) {
    return this.disc.position.clone().setY(0).addScaledVector(this.right, x).addScaledVector(UP, y).addScaledVector(this.fwd, z);
  }
}
