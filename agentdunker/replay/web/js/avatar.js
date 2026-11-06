// Visual-only adapter for the user's Holo Court v2 Xbot asset.
// Upstream: c0ce80a9ef0f2b22261ec9aa9e9c07eaac039aec, js/avatar.js.
// Xbot.glb SHA256: 002f8d269de68e5dce3d25195caf390d1aa359bbfaae3fcf4c8dc78ec36c3ba5.
// Model origin in upstream README: Three.js examples / Mixamo Xbot.
// Skin point sampling/shaders derive from that upstream implementation; rig
// adaptation below consumes CourtVision's final constrained joints, not motion
// observations. No timeline, contact solver, audio or source-data mutation here.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { clone } from 'three/addons/utils/SkeletonUtils.js';

let template = null;
let templateKey = null;
const sampled = new WeakMap();
const UP = new THREE.Vector3(0, 1, 0);
const X = new THREE.Vector3(1, 0, 0);
const Z = new THREE.Vector3(0, 0, 1);
const SIDES = [['L', 'Left'], ['R', 'Right']];
const FINGERS = ['Index', 'Middle', 'Ring', 'Pinky'];
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const finite = v => v?.isVector3 && [v.x, v.y, v.z].every(Number.isFinite);
const unit = (v, fallback = UP) => v.lengthSq() > 1e-12 ? v.normalize() : v.copy(fallback).normalize();

export async function loadAvatar(url) {
  if (template && templateKey === url) return template;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Avatar load failed (${response.status}): ${url}`);
  const buffer = await response.arrayBuffer();
  const resourcePath = /^data:/.test(url) ? '' : new URL('.', new URL(url, globalThis.location?.href)).href;
  const gltf = await new GLTFLoader().parseAsync(buffer, resourcePath);
  template = gltf.scene;
  templateKey = url;
  return template;
}

// All fallbacks depend on this pose and fixed axes, never on the last render.
function frame(along, preferred, fallback = X) {
  const y = unit(along.clone());
  const x = preferred.clone().addScaledVector(y, -preferred.dot(y));
  if (x.lengthSq() < 1e-10) x.copy(fallback).addScaledVector(y, -fallback.dot(y));
  if (x.lengthSq() < 1e-10) {
    const axis = Math.abs(y.x) < .8 ? X : Z;
    x.copy(axis).addScaledVector(y, -axis.dot(y));
  }
  x.normalize();
  return new THREE.Matrix4().makeBasis(x, y, new THREE.Vector3().crossVectors(x, y));
}

function transported(lateral, from, to) {
  const a = unit(from.clone()), b = unit(to.clone());
  const q = a.dot(b) < -.999999
    ? new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3().setFromMatrixColumn(frame(a, lateral), 0), Math.PI)
    : new THREE.Quaternion().setFromUnitVectors(a, b);
  return lateral.clone().applyQuaternion(q);
}

function floorHeight(value, floor, width) {
  const h=value-floor;
  if(h>=width)return value;if(h<=-width)return floor;
  return floor+(h+width)**2/(4*width);
}

// The existing controller may place a reconstructed ankle under the court;
// its procedural skin projects the visible surface above the floor. A GLB
// needs an equivalent presentation constraint. Re-bend the visual leg without
// altering the controller, hips, arm contacts or source observations.
function groundedLeg(hip, knee, ankle, clearance, forward) {
  const a=hip,b=ankle.clone(),l1=hip.distanceTo(knee),l2=knee.distanceTo(ankle);
  b.y=floorHeight(b.y,clearance,.015);
  const axis=b.clone().sub(a),requested=axis.length();unit(axis,new THREE.Vector3(0,-1,0));
  const distance=clamp(requested,Math.abs(l1-l2)+1e-6,l1+l2-1e-6);
  b.copy(a).addScaledVector(axis,distance);
  const pole=knee.clone().sub(a).addScaledVector(axis,-knee.clone().sub(a).dot(axis));
  if(pole.lengthSq()<1e-10)pole.copy(forward).addScaledVector(axis,-forward.dot(axis));
  unit(pole,new THREE.Vector3(1,0,0));
  const along=(l1*l1-l2*l2+distance*distance)/(2*distance),height=Math.sqrt(Math.max(0,l1*l1-along*along));
  return {knee:a.clone().addScaledVector(axis,along).addScaledVector(pole,height),ankle:b,correction:b.distanceTo(ankle)};
}

function material(color, joint) {
  const m = new THREE.MeshStandardMaterial({
    color: color.clone().multiplyScalar(joint ? .34 : .46),
    emissive: color, emissiveIntensity: joint ? .045 : .09,
    metalness: .18, roughness: .5,
  });
  m.onBeforeCompile = shader => {
    shader.uniforms.uRim = {value: color.clone().multiplyScalar(joint ? .48 : .85)};
    shader.fragmentShader = `uniform vec3 uRim;\n${shader.fragmentShader}`.replace(
      '#include <emissivemap_fragment>',
      '#include <emissivemap_fragment>\nfloat rim = pow(1.0-abs(dot(normalize(vViewPosition),normal)),2.4);\ntotalEmissiveRadiance += uRim*rim;',
    );
  };
  m.customProgramCacheKey = () => `courtvision-xbot-${joint ? 'joint' : 'body'}`;
  return m;
}

const POINT_VS = /* glsl */ `
  #include <common>
  #include <skinning_pars_vertex>
  attribute float seed;
  attribute vec3 nrm;
  uniform float uTime, uSize, uScale;
  varying float vAlpha;
  void main() {
    vec3 transformed = position;
    vec3 objectNormal = nrm;
    #include <skinbase_vertex>
    #include <skinnormal_vertex>
    #include <skinning_vertex>
    vec4 wp = modelMatrix * vec4(transformed,1.0);
    vec3 n = normalize(mat3(modelMatrix)*objectNormal);
    float ndv = dot(n,normalize(cameraPosition-wp.xyz));
    float scan = .82+.18*sin(wp.y*95.0-uTime*3.0);
    vAlpha = (.18+.75*pow(1.0-abs(ndv),1.7))*(ndv>0.0?1.0:.35)*scan*(.94+.06*sin(uTime*7.0+seed*40.0));
    vec4 mv = viewMatrix*wp;
    gl_PointSize = uSize*uScale/-mv.z;
    gl_Position = projectionMatrix*mv;
  }`;
const POINT_FS = /* glsl */ `
  uniform vec3 uColor;
  varying float vAlpha;
  void main() {
    float d = length(gl_PointCoord-.5);
    if(d>.5)discard;
    gl_FragColor=vec4(uColor*(1.15-d),(1.0-d*1.8)*vAlpha);
  }`;

// Independent seed per mesh. Construction order, other random consumers and
// browser refreshes cannot change the model's point positions or skin weights.
function samplePoints(geo, count, name) {
  let state = 0x58424f54;
  for (const c of name) state = Math.imul(state ^ c.charCodeAt(0), 16777619) >>> 0;
  const random = () => ((state = (Math.imul(1664525, state) + 1013904223) >>> 0) / 4294967296);
  const P = geo.attributes.position, N = geo.attributes.normal;
  const SI = geo.attributes.skinIndex, SW = geo.attributes.skinWeight;
  const idx = geo.index?.array ?? Uint32Array.from({length:P.count}, (_, i) => i);
  const triangles = idx.length / 3, cdf = new Float64Array(triangles);
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  let total = 0;
  for (let t = 0; t < triangles; t++) {
    a.fromBufferAttribute(P,idx[3*t]);b.fromBufferAttribute(P,idx[3*t+1]);c.fromBufferAttribute(P,idx[3*t+2]);
    total += b.sub(a).cross(c.sub(a)).length()/2;cdf[t]=total;
  }
  const pos=new Float32Array(count*3),nrm=new Float32Array(count*3),skinIndex=new Uint16Array(count*4),skinWeight=new Float32Array(count*4),seed=new Float32Array(count);
  const v=new THREE.Vector3(),n=new THREE.Vector3(),tmp=new THREE.Vector3();
  for(let k=0;k<count;k++){
    const r=(k+random())*total/count;let lo=0,hi=triangles-1;
    while(lo<hi){const mid=(lo+hi)>>1;if(cdf[mid]<r)lo=mid+1;else hi=mid;}
    let u=random(),w=random();if(u+w>1){u=1-u;w=1-w;}
    const bary=[1-u-w,u,w],weights=new Map();v.set(0,0,0);n.set(0,0,0);
    for(let j=0;j<3;j++){
      const vi=idx[lo*3+j];v.addScaledVector(tmp.fromBufferAttribute(P,vi),bary[j]);n.addScaledVector(tmp.fromBufferAttribute(N,vi),bary[j]);
      for(let q=0;q<4;q++){const weight=SW.getComponent(vi,q)*bary[j],bone=SI.getComponent(vi,q);if(weight>0)weights.set(bone,(weights.get(bone)??0)+weight);}
    }
    pos.set(v.toArray(),k*3);nrm.set(n.normalize().toArray(),k*3);
    const top=[...weights].sort((x,y)=>y[1]-x[1]).slice(0,4),sum=top.reduce((s,[,x])=>s+x,0)||1;
    top.forEach(([bone,weight],q)=>{skinIndex[k*4+q]=bone;skinWeight[k*4+q]=weight/sum;});seed[k]=random();
  }
  const geometry=new THREE.BufferGeometry();
  for(const [key,array,size] of [['position',pos,3],['nrm',nrm,3],['skinIndex',skinIndex,4],['skinWeight',skinWeight,4],['seed',seed,1]])geometry.setAttribute(key,new THREE.BufferAttribute(array,size));
  return geometry;
}

export class AvatarPlayer {
  constructor(color, shared, {style='xbot-points',heightMeters=1.88}={}) {
    if(!template)throw new Error('Call and await loadAvatar before creating AvatarPlayer');
    if(!['xbot-points','xbot-mesh'].includes(style))throw new Error(`Unsupported avatar style: ${style}`);
    this.style=style;this.color=new THREE.Color(color);
    this.bodyScale=Number.isFinite(heightMeters)?clamp(heightMeters/1.88,.85,1.25):1;
    this.heightMeters=1.88*this.bodyScale;
    this.root=clone(template);this.group=new THREE.Group();this.group.add(this.root);
    this.bones={};this.skeletons=[];this.pointGeometries=[];
    this.root.traverse(o=>{if(o.isBone)this.bones[o.name.replace(/^mixamorig:?/,'')]=o;});
    this.group.updateMatrixWorld(true);
    const B=this.bones,wp=name=>B[name].getWorldPosition(new THREE.Vector3());
    this.rest={};
    for(const [name,bone] of Object.entries(B))this.rest[name]={world:bone.matrixWorld.clone(),local:bone.matrix.clone(),position:wp(name)};
    const modelHeight=new THREE.Box3().setFromObject(this.root).getSize(new THREE.Vector3()).y;
    this.widthScale=this.heightMeters/modelHeight;
    this.footClearance={L:0,R:0};
    this.root.traverse(mesh=>{
      if(!mesh.isSkinnedMesh)return;
      const skinIndex=mesh.geometry.attributes.skinIndex,skinWeight=mesh.geometry.attributes.skinWeight;
      const position=mesh.geometry.attributes.position;
      for(let i=0;i<position.count;i++)for(const [side,prefix] of SIDES){
        let weight=0;
        for(let k=0;k<4;k++)if(new RegExp(`${prefix}(Foot|Toe)`).test(mesh.skeleton.bones[skinIndex.getComponent(i,k)].name))weight+=skinWeight.getComponent(i,k);
        if(weight<.5)continue;
        const point=new THREE.Vector3().fromBufferAttribute(position,i).applyMatrix4(mesh.matrixWorld);
        this.footClearance[side]=Math.max(this.footClearance[side],wp(prefix+'Foot').y-point.y);
      }
    });
    for(const side of ['L','R'])this.footClearance[side]=this.footClearance[side]*this.widthScale+.014*this.bodyScale;
    this.restRight=unit(wp('RightArm').sub(wp('LeftArm')),X);
    this.restForward=unit(new THREE.Vector3().crossVectors(UP,this.restRight),Z);
    this.handRest={};
    for(const [side,prefix] of SIDES){
      const wrist=wp(prefix+'Hand'),mcp=new THREE.Vector3();
      for(const f of FINGERS)mcp.add(wp(prefix+'Hand'+f+'1'));
      mcp.multiplyScalar(.25);
      const offset=mcp.clone().sub(wrist).multiplyScalar(.5),along=unit(offset.clone());
      const across=unit(new THREE.Vector3().crossVectors(along,UP),this.restForward);
      this.handRest[side]={offset,frame:frame(along,across),localAcross:across.clone().transformDirection(this.rest[prefix+'Hand'].world.clone().invert())};
      // Fixed relaxed fingers, not inferred finger motion. Keep MCP translation
      // unchanged so the actual model's palm reference remains measurable.
      for(const f of FINGERS)for(let k=1;k<=3;k++){
        const name=prefix+'Hand'+f+k,bone=B[name];if(!bone)continue;
        const pos=new THREE.Vector3(),quat=new THREE.Quaternion(),scale=new THREE.Vector3();
        this.rest[name].local.decompose(pos,quat,scale);
        const axis=across.clone().transformDirection(this.rest[name].world.clone().invert());
        quat.multiply(new THREE.Quaternion().setFromAxisAngle(axis,-.24));
        this.rest[name].local.compose(pos,quat,scale);
      }
    }
    this.root.traverse(o=>{
      if(o.isBone)o.matrixAutoUpdate=false;
      if(!o.isSkinnedMesh)return;
      o.frustumCulled=false;const joint=/Joints/i.test(o.name);
      if(!this.skeletons.includes(o.skeleton))this.skeletons.push(o.skeleton);
      if(style==='xbot-mesh'){o.material=material(this.color,joint);return;}
      o.material=new THREE.MeshBasicMaterial({colorWrite:false,polygonOffset:true,polygonOffsetFactor:2,polygonOffsetUnits:8});
      let geometry=sampled.get(o.geometry);
      if(!geometry){geometry=samplePoints(o.geometry,joint?4500:26000,o.name);sampled.set(o.geometry,geometry);}
      this.pointGeometries.push(geometry);o.skeleton.computeBoneTexture();
      const points=new THREE.Points(geometry,new THREE.ShaderMaterial({
        vertexShader:POINT_VS,fragmentShader:POINT_FS,
        uniforms:{uColor:{value:this.color.clone().multiplyScalar(joint?.60:1)},uTime:shared.uTime,uScale:shared.uScale,uSize:{value:.018},
          bindMatrix:{value:o.bindMatrix},bindMatrixInverse:{value:o.bindMatrixInverse},boneTexture:{value:o.skeleton.boneTexture}},
        defines:{USE_SKINNING:''},transparent:true,depthWrite:false,blending:THREE.AdditiveBlending,
      }));points.frustumCulled=false;o.add(points);
    });
    // Xbot's original rigid torso plates stop short of the neck/waist. The
    // controller has different basketball proportions, so those design seams
    // can expose empty space. Two small continuous under-skin transitions join
    // the existing plates; the user's model and contact skeleton stay intact.
    this.transitions=[this._transition('waist',28,28,shared),this._transition('neck',18,20,shared)];
    this.joints={headBase:new THREE.Vector3(),handL:new THREE.Vector3(),handR:new THREE.Vector3()};
    this.fwd=new THREE.Vector3();this.right=new THREE.Vector3();this.diagnostics={};
  }

  _transition(name,rings,sides,shared) {
    const count=(rings+1)*(sides+1),position=new Float32Array(count*3),normal=new Float32Array(count*3),seed=new Float32Array(count),indices=[];
    for(let r=0;r<=rings;r++)for(let a=0;a<=sides;a++){
      const i=r*(sides+1)+a;seed[i]=((Math.imul(i+1,2654435761)>>>0)%65536)/65536;
      if(r<rings&&a<sides)indices.push(i,i+sides+1,i+1,i+1,i+sides+1,i+sides+2);
    }
    const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(position,3));geometry.setAttribute('normal',new THREE.BufferAttribute(normal,3));geometry.setAttribute('nrm',new THREE.BufferAttribute(normal,3));geometry.setAttribute('seed',new THREE.BufferAttribute(seed,1));geometry.setIndex(indices);
    const mesh=new THREE.Mesh(geometry,this.style==='xbot-mesh'?material(this.color,false):new THREE.MeshBasicMaterial({colorWrite:false,polygonOffset:true,polygonOffsetFactor:2,polygonOffsetUnits:8}));
    mesh.frustumCulled=false;mesh.name=`CourtVision_${name}_transition`;this.group.add(mesh);
    let pointGeometry=null,samples=[];
    if(this.style==='xbot-points'){
      const number=name==='waist'?1400:300,pointPosition=new Float32Array(number*3),pointNormal=new Float32Array(number*3),pointSeed=new Float32Array(number);
      let state=name==='waist'?0x57414953:0x4e45434b;const random=()=>((state=(Math.imul(1664525,state)+1013904223)>>>0)/4294967296);
      for(let i=0;i<number;i++){samples.push([(i+random())/number,random()]);pointSeed[i]=random();}
      pointGeometry=new THREE.BufferGeometry();pointGeometry.setAttribute('position',new THREE.BufferAttribute(pointPosition,3));pointGeometry.setAttribute('nrm',new THREE.BufferAttribute(pointNormal,3));pointGeometry.setAttribute('seed',new THREE.BufferAttribute(pointSeed,1));
      const points=new THREE.Points(pointGeometry,new THREE.ShaderMaterial({vertexShader:POINT_VS,fragmentShader:POINT_FS,
        uniforms:{uColor:{value:this.color},uTime:shared.uTime,uScale:shared.uScale,uSize:{value:.014}},
        transparent:true,depthWrite:false,blending:THREE.AdditiveBlending}));points.frustumCulled=false;points.name=mesh.name+'_points';this.group.add(points);
    }
    return{name,rings,sides,geometry,position,normal,pointGeometry,samples};
  }

  _updateTransitions(control) {
    const J=control.joints,s=this.bodyScale,local=this.group.matrixWorld.clone().invert();
    const neckAxis=unit(J.headBase.clone().sub(J.neck));
    for(const part of this.transitions){
      const waist=part.name==='waist';
      const start=waist?J.pelvis.clone():J.neck.clone().lerp(J.chest,.28);
      const end=waist?J.spine.clone().lerp(J.chest,.5):J.headBase.clone().addScaledVector(neckAxis,.045*s);
      const middle=waist?J.spine.clone():start.clone().lerp(end,.5);
      for(let r=0;r<=part.rings;r++){
        const u=r/part.rings,center=start.clone().multiplyScalar((1-u)**2).addScaledVector(middle,2*u*(1-u)).addScaledVector(end,u*u);
        const tangent=middle.clone().sub(start).multiplyScalar(2*(1-u)).addScaledVector(end.clone().sub(middle),2*u);
        const basis=frame(tangent,this.right,this.fwd),right=new THREE.Vector3().setFromMatrixColumn(basis,0),forward=new THREE.Vector3().setFromMatrixColumn(basis,2);
        const radius=(waist?(.122-.017*Math.sin(Math.PI*u)+.031*u*u):(.050-.004*Math.sin(Math.PI*u)))*s;
        const depth=radius*(waist?.78:.94);
        for(let a=0;a<=part.sides;a++){
          const theta=2*Math.PI*a/part.sides,c=Math.cos(theta),sn=Math.sin(theta),i=(r*(part.sides+1)+a)*3;
          const point=center.clone().addScaledVector(right,radius*c).addScaledVector(forward,depth*sn).applyMatrix4(local);
          const normal=right.clone().multiplyScalar(c/radius).addScaledVector(forward,sn/depth).normalize().transformDirection(local);
          part.position.set(point.toArray(),i);part.normal.set(normal.toArray(),i);
        }
      }
      part.geometry.attributes.position.needsUpdate=true;part.geometry.attributes.normal.needsUpdate=true;part.geometry.attributes.nrm.needsUpdate=true;
      if(part.pointGeometry){
        const points=part.pointGeometry.attributes.position,normals=part.pointGeometry.attributes.nrm;
        part.samples.forEach(([u,v],i)=>{
          const fr=u*part.rings,fa=v*part.sides,r=Math.min(part.rings-1,Math.floor(fr)),a=Math.floor(fa),dr=fr-r,da=fa-a;
          const corners=[[r,a,(1-dr)*(1-da)],[r,a+1,(1-dr)*da],[r+1,a,dr*(1-da)],[r+1,a+1,dr*da]];
          for(let c=0;c<3;c++){points.array[i*3+c]=0;normals.array[i*3+c]=0;
            for(const [rr,aa,w] of corners){const k=(rr*(part.sides+1)+aa)*3+c;points.array[i*3+c]+=part.position[k]*w;normals.array[i*3+c]+=part.normal[k]*w;}
          }
        });points.needsUpdate=true;normals.needsUpdate=true;
      }
    }
  }

  _worldMatrix(name, world) {
    const bone=this.bones[name];
    // Explicit affine local matrices avoid inherited nonuniform-scale shear:
    // each final world segment maps its own rest endpoints to rig endpoints.
    bone.matrix.copy(bone.parent.matrixWorld).invert().multiply(world);
    bone.matrixWorldNeedsUpdate=true;bone.updateMatrixWorld(true);
  }

  _deform(name, restFrame, targetFrame, origin, alongScale, transverseScale=this.widthScale) {
    const transform=new THREE.Matrix4().makeTranslation(...origin.toArray())
      .multiply(targetFrame).multiply(new THREE.Matrix4().makeScale(transverseScale,alongScale,transverseScale))
      .multiply(restFrame.clone().transpose())
      .multiply(new THREE.Matrix4().makeTranslation(...this.rest[name].position.clone().negate().toArray()));
    this._worldMatrix(name,transform.multiply(this.rest[name].world));
  }

  _segment(name, child, a, b, lateral, restLateral=this.restRight) {
    const restAxis=this.rest[child].position.clone().sub(this.rest[name].position);
    const axis=b.clone().sub(a),restLength=restAxis.length();
    if(axis.lengthSq()<1e-12)axis.copy(UP).multiplyScalar(restLength*this.widthScale);
    this._deform(name,frame(restAxis,restLateral,this.restForward),frame(axis,lateral,this.right),a,axis.length()/restLength);
  }

  updateFromRig(control) {
    const J=control.joints;
    for(const name of ['pelvis','spine','chest','neck','headBase','shoulderL','shoulderR','elbowL','elbowR','handL','handR','hipL','hipR','kneeL','kneeR','ankleL','ankleR'])
      if(!finite(J?.[name]))throw new Error(`Avatar rig joint must be finite: ${name}`);
    this.right.copy(finite(control.right)?control.right:X);unit(this.right,X);
    this.fwd.copy(finite(control.fwd)?control.fwd:Z);unit(this.fwd,Z);
    for(const [name,bone] of Object.entries(this.bones)){bone.matrix.copy(this.rest[name].local);bone.matrixWorldNeedsUpdate=true;}
    this.group.updateMatrixWorld(true);
    const low=J.pelvis.clone().lerp(J.spine,.5);
    this._segment('Hips','Spine',J.pelvis,low,this.right);
    this._segment('Spine','Spine1',low,J.spine,this.right);
    this._segment('Spine1','Spine2',J.spine,J.chest,this.right);
    this._segment('Spine2','Neck',J.chest,J.neck,this.right);
    this._segment('Neck','Head',J.neck,J.headBase,this.right);
    const headAxis=unit(J.headBase.clone().sub(J.neck));
    this._segment('Head','HeadTop_End',J.headBase,J.headBase.clone().addScaledVector(headAxis,.18*this.bodyScale),this.right);
    const errors=[],legCorrections=[];
    for(const [side,prefix] of SIDES){
      const shoulder=J['shoulder'+side],elbow=J['elbow'+side],wrist=J['hand'+side];
      const collar=J.chest.clone().lerp(shoulder,.25);
      this._segment(prefix+'Shoulder',prefix+'Arm',collar,shoulder,this.fwd,this.restForward);
      const guide=(kind,axis)=>{
        const cached=control.appearanceFrames?.sample(control.gesture?.time,kind+side);
        return cached&&finite(cached.axis)&&finite(cached.lateral)
          ?transported(cached.lateral,cached.axis,axis):this.fwd.clone();
      };
      this._segment(prefix+'Arm',prefix+'ForeArm',shoulder,elbow,guide('arm',elbow.clone().sub(shoulder)),this.restForward);
      this._segment(prefix+'ForeArm',prefix+'Hand',elbow,wrist,guide('arm',wrist.clone().sub(elbow)),this.restForward);
      const hand=control.getHandFrame(side);
      if(!hand||!finite(hand.palmCenter)||!finite(hand.along)||!finite(hand.across))throw new Error(`Avatar requires final palm frame: ${side}`);
      const rest=this.handRest[side],offset=hand.palmCenter.clone().sub(wrist);
      this._deform(prefix+'Hand',rest.frame,frame(offset,hand.across,this.fwd),wrist,offset.length()/rest.offset.length());
      const leg=groundedLeg(J['hip'+side],J['knee'+side],J['ankle'+side],this.footClearance[side],this.fwd);
      legCorrections.push({side,ankleCorrectionMeters:leg.correction,sourceAnkleY:J['ankle'+side].y,renderedAnkleY:leg.ankle.y});
      this._segment(prefix+'UpLeg',prefix+'Leg',J['hip'+side],leg.knee,guide('leg',leg.knee.clone().sub(J['hip'+side])));
      this._segment(prefix+'Leg',prefix+'Foot',leg.knee,leg.ankle,guide('leg',leg.ankle.clone().sub(leg.knee)));
      const restToe=this.rest[prefix+'ToeBase'].position.clone().sub(this.rest[prefix+'Foot'].position);
      const toe=finite(J['toeDir'+side])?J['toeDir'+side].clone():this.fwd.clone();toe.y=0;unit(toe,this.fwd);
      const restForward=restToe.clone().setY(0);unit(restForward,this.restForward);
      this._deform(prefix+'Foot',frame(restForward,UP),frame(toe,UP),leg.ankle,this.widthScale);
      this.group.updateMatrixWorld(true);
      this.joints['hand'+side].copy(this.bones[prefix+'Hand'].getWorldPosition(new THREE.Vector3()));
      errors.push({side,wristErrorMeters:this.joints['hand'+side].distanceTo(wrist),palmErrorMeters:this.getPalmPosition(side).distanceTo(hand.palmCenter)});
    }
    this.group.updateMatrixWorld(true);
    for(const skeleton of this.skeletons)skeleton.update();
    this._updateTransitions(control);
    this.joints.headBase.copy(this.bones.Head.getWorldPosition(new THREE.Vector3()));
    this.diagnostics={style:this.style,headErrorMeters:this.joints.headBase.distanceTo(J.headBase),hands:errors,legCorrections,
      palmReference:'midpoint of actual wrist and four actual MCP bone origins',historyIndependent:true};
    return this;
  }

  getHandFrame(side='R') {
    const prefix=side==='L'?'Left':'Right',bone=this.bones[prefix+'Hand'];
    const wrist=bone.getWorldPosition(new THREE.Vector3()),mcp=new THREE.Vector3();
    for(const f of FINGERS)mcp.add(this.bones[prefix+'Hand'+f+'1'].getWorldPosition(new THREE.Vector3()));
    mcp.multiplyScalar(.25);const palmCenter=wrist.clone().lerp(mcp,.5);
    const along=unit(palmCenter.clone().sub(wrist)),across=this.handRest[side==='L'?'L':'R'].localAcross.clone().transformDirection(bone.matrixWorld);
    across.addScaledVector(along,-across.dot(along));unit(across,this.right);
    const depth=new THREE.Vector3().crossVectors(across,along).normalize();
    return{wrist,palmCenter,along,across,depth,normal:depth.clone().negate(),bodyScale:this.bodyScale};
  }

  getPalmPosition(side='R') { return this.getHandFrame(side).palmCenter; }
}
