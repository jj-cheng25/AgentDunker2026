// Presentation only: immutable whole-window actor bounds and a stable camera.
import * as THREE from 'three';

const finite3 = p => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite);
const UP = new THREE.Vector3(0, 1, 0);

function corners(box) {
  const result = [];
  for (const x of [box.min.x, box.max.x])
    for (const y of [box.min.y, box.max.y])
      for (const z of [box.min.z, box.max.z]) result.push(new THREE.Vector3(x,y,z));
  return result;
}

export function actorWindowEnvelope(play) {
  const coverage = play.metadata?.sourceCoverage;
  const start = coverage?.startSec ?? 0, stop = coverage?.stopSec ?? play.duration;
  if (![start,stop].every(Number.isFinite) || stop <= start) throw new Error('Invalid actor framing source window');
  const box = new THREE.Box3(), actors = [];
  for (const actor of play.players) {
    const body = new THREE.Box3(), height = play.motion?.players?.[actor.id]?.heightMeters ?? actor.heightMeters ?? 1.88;
    if (!Number.isFinite(height) || height <= 0) throw new Error('Invalid actor framing height');
    const radius = height*.65, ceiling = height+Math.max(1.,height*.5);
    const path = actor.path ?? [], positions = [];
    for (const row of path) if (finite3(row) && start <= row[0] && row[0] <= stop) positions.push(row);
    for (const time of [start,stop]) {
      for (let i=0;i<path.length-1;i++) {
        const a=path[i],b=path[i+1];
        if (a[0] <= time && time <= b[0] && b[0] > a[0]) {
          const u=(time-a[0])/(b[0]-a[0]);
          positions.push([time,a[1]+u*(b[1]-a[1]),a[2]+u*(b[2]-a[2])]);break;
        }
      }
    }
    if (!positions.length) throw new Error('Actor has no path in observed framing window: '+actor.id);
    for (const [,x,z] of positions) {
      body.expandByPoint(new THREE.Vector3(x-radius,-.25,z-radius));
      body.expandByPoint(new THREE.Vector3(x+radius,ceiling,z+radius));
    }
    let motionSamples=0;
    for (const sample of play.motion?.players?.[actor.id]?.samples ?? []) {
      if (!(start <= sample.t && sample.t <= stop)) continue;
      for (const joint of sample.jointsWorld ?? []) if (finite3(joint)) {
        const p=new THREE.Vector3(...joint),pad=new THREE.Vector3().setScalar(height*.20);
        body.expandByPoint(p.clone().sub(pad));body.expandByPoint(p.clone().add(pad));
      }
      motionSamples++;
    }
    box.union(body);
    actors.push({id:actor.id,min:body.min.toArray(),max:body.max.toArray(),pathPoints:positions.length,motionSamples});
  }
  if (box.isEmpty()) throw new Error('No actors to frame');
  return {box,sourceWindow:[start,stop],actors};
}

export function actorSafeRect(width,height,{panelWidth=0,captions=false}={}) {
  const rect={left:24,top:76,right:width-24-panelWidth,bottom:height-(captions?112:32)};
  if (rect.right-rect.left < 120 || rect.bottom-rect.top < 120) throw new Error('Insufficient actor framing viewport');
  return {...rect,width:rect.right-rect.left,height:rect.bottom-rect.top};
}

export function fitActorCamera(envelope,preset,{width,height,safeRect,fov=40,orbitRadians=0,scale=1}={}) {
  if (![width,height,fov,orbitRadians].every(Number.isFinite) || width<=0 || height<=0)
    throw new Error('Invalid actor framing viewport/orbit');
  const target=envelope.box.getCenter(new THREE.Vector3());
  const outward=new THREE.Vector3(...preset.pos).sub(new THREE.Vector3(...preset.target)).normalize();
  // A union's outer AABB introduces corners occupied by no actor. Use every
  // actor's complete-window conservative box instead, retaining the same
  // global target and coverage without framing those fictitious corners.
  const points=envelope.actors.flatMap(actor=>corners(new THREE.Box3(
    new THREE.Vector3(...actor.min),new THREE.Vector3(...actor.max)))).map(p=>p.sub(target));
  const radius=Math.max(...points.map(p=>p.length()));
  const tangent=Math.tan(THREE.MathUtils.degToRad(fov/2));
  const tanX=tangent*(width/height)*(safeRect.width/width),tanY=tangent*(safeRect.height/height);
  const span=Math.abs(orbitRadians),steps=Math.max(1,Math.ceil(span/.015));
  let distance=0;
  for(let i=0;i<=steps;i++) {
    const direction=outward.clone().applyAxisAngle(UP,span*(i/steps-.5));
    const right=new THREE.Vector3().crossVectors(UP,direction).normalize();
    if(right.lengthSq()<.5)right.set(1,0,0);
    const up=new THREE.Vector3().crossVectors(direction,right).normalize();
    for(const p of points) distance=Math.max(distance,p.dot(direction)+Math.abs(p.dot(right))/tanX,
                                           p.dot(direction)+Math.abs(p.dot(up))/tanY);
  }
  // Every orientation is within half a sample step. This derivative bound
  // covers the continuous orbit between tested angles, without frame zoom.
  const angularGuard=radius*(1+1/Math.min(tanX,tanY))*(span/steps)/2;
  // scale < 1 trades a conservative whole-window bound for a closer, reference-like framing.
  distance=(Math.max(distance+angularGuard,radius+.2)*1.035+.1)*scale;
  return {target:target.toArray(),outward:outward.toArray(),distance,radius,
    viewport:{width,height},safeRect,sourceWindow:envelope.sourceWindow,
    worldBounds:{min:envelope.box.min.toArray(),max:envelope.box.max.toArray()},
    actorCount:envelope.actors.length,orbitRadians,angularGuardMeters:angularGuard,
    method:'whole-observed-window-per-actor-envelopes-v2',supportCornerCount:points.length,timeDependentZoom:false,
    offset:{x:width/2-(safeRect.left+safeRect.right)/2,y:height/2-(safeRect.top+safeRect.bottom)/2}};
}

export function applyActorCamera(camera,controls,fit,angle) {
  const target=new THREE.Vector3(...fit.target);
  const offset=new THREE.Vector3(...fit.outward).applyAxisAngle(UP,angle).multiplyScalar(fit.distance);
  camera.setViewOffset(fit.viewport.width,fit.viewport.height,fit.offset.x,fit.offset.y,fit.viewport.width,fit.viewport.height);
  camera.far=Math.max(camera.far,fit.distance+fit.radius+20);
  camera.updateProjectionMatrix();
  controls.maxDistance=Math.max(controls.maxDistance,fit.distance+1);
  camera.position.copy(target).add(offset);controls.target.copy(target);controls.update();camera.updateMatrixWorld(true);
}

function visible(object) {
  for(let current=object;current;current=current.parent)if(!current.visible)return false;
  return true;
}

export function actorScreenBounds(group,camera,{width,height,safeRect=null,panels=[],pixelRatio=1}={}) {
  group.updateMatrixWorld(true);camera.updateMatrixWorld(true);
  const low=[Infinity,Infinity],high=[-Infinity,-Infinity];
  let vertices=0,behind=0,depthClipped=0;
  const point=new THREE.Vector3(),base=new THREE.Vector3(),temp=new THREE.Vector3(),bone=new THREE.Matrix4();
  group.traverse(object=>{
    const position=object.geometry?.attributes?.position;
    if(!position || !visible(object) || !(object.isMesh || object.isPoints))return;
    const skin=object.isPoints && object.material?.defines?.USE_SKINNING !== undefined && object.parent?.isSkinnedMesh
      ? object.parent:null;
    if(object.isSkinnedMesh)object.skeleton.update();
    if(skin)skin.skeleton.update();
    for(let i=0;i<position.count;i++) {
      if(object.isSkinnedMesh)object.getVertexPosition(i,point);
      else if(skin) {
        base.fromBufferAttribute(position,i).applyMatrix4(skin.bindMatrix);point.set(0,0,0);
        const indices=object.geometry.attributes.skinIndex,weights=object.geometry.attributes.skinWeight;
        for(let j=0;j<4;j++) {
          const weight=weights.getComponent(i,j);if(!weight)continue;
          bone.fromArray(skin.skeleton.boneMatrices,indices.getComponent(i,j)*16);
          point.addScaledVector(temp.copy(base).applyMatrix4(bone),weight);
        }
        point.applyMatrix4(skin.bindMatrixInverse);
      } else point.fromBufferAttribute(position,i);
      point.applyMatrix4(object.matrixWorld);
      const depth=-temp.copy(point).applyMatrix4(camera.matrixWorldInverse).z;
      vertices++;
      if(depth<=0){behind++;continue;}
      point.project(camera);
      if(point.z < -1 || point.z > 1)depthClipped++;
      const x=(point.x+1)*width/2,y=(1-point.y)*height/2;
      const uniforms=object.material?.uniforms;
      const sprite=object.isPoints && uniforms?.uSize && uniforms?.uScale
        ? uniforms.uSize.value*uniforms.uScale.value/depth/(2*pixelRatio):0;
      const padding=4+sprite; // include raster point radius and a small bloom guard
      low[0]=Math.min(low[0],x-padding);low[1]=Math.min(low[1],y-padding);
      high[0]=Math.max(high[0],x+padding);high[1]=Math.max(high[1],y+padding);
    }
  });
  if(!low.every(Number.isFinite) || !high.every(Number.isFinite))
    return {method:'cpu-projected-render-geometry-v1',vertices,valid:false,fullyInsideViewport:false};
  const bbox={left:low[0],top:low[1],right:high[0],bottom:high[1],width:high[0]-low[0],height:high[1]-low[1]};
  const contains=r=>!behind && !depthClipped && bbox.left>=r.left && bbox.top>=r.top && bbox.right<=r.right && bbox.bottom<=r.bottom;
  const clippedEdges=[];
  if(bbox.left<0)clippedEdges.push('left');if(bbox.top<0)clippedEdges.push('top');
  if(bbox.right>width)clippedEdges.push('right');if(bbox.bottom>height)clippedEdges.push('bottom');
  if(behind || depthClipped)clippedEdges.push('depth');
  return {method:'cpu-projected-render-geometry-v1',valid:true,bbox,vertices,behindCameraVertices:behind,
    depthClippedVertices:depthClipped,fullyInsideViewport:contains({left:0,top:0,right:width,bottom:height}),
    fullyInsideSafeRect:safeRect?contains(safeRect):null,clippedEdges,
    overlapsPanels:panels.filter(r=>r && bbox.left<r.right && bbox.right>r.x && bbox.top<r.bottom && bbox.bottom>r.y).map(r=>r.name),
    minimumViewportMarginPixels:Math.min(bbox.left,bbox.top,width-bbox.right,height-bbox.bottom)};
}
