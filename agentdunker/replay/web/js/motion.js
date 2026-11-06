// Source-clock sampling of recovered motion. This never extends observations
// through an unbounded gap and keeps provenance/coverage available to the body.
const SMOOTH_CONFIDENCE = 0.3;
const smooth01 = (x) => { const y = Math.max(0, Math.min(1, x)); return y*y*(3-2*y); };

// Local PCHIP slopes retain the observations and preserve each coordinate's
// range. Unlike unconstrained Catmull-Rom, an occlusion/reversal cannot fling a
// limb beyond its two source samples. Slopes depend only on nearby source data,
// so seeking, playing backwards and rendering frames out of order agree.
function reliableJoint(sample, joint) {
  return sample?.observed !== false &&
    (sample?.confidence?.[joint] ?? 0) >= SMOOTH_CONFIDENCE &&
    sample?.jointsWorld?.[joint]?.every(Number.isFinite);
}

function endpointSlope(h0, h1, d0, d1) {
  const slope = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
  if (Math.sign(slope) !== Math.sign(d0)) return 0;
  if (Math.sign(d0) !== Math.sign(d1) && Math.abs(slope) > 3 * Math.abs(d0)) return 3 * d0;
  return slope;
}

function jointSlope(samples, index, joint, axis, maxGap) {
  const centre = samples[index];
  const usable = (other) => reliableJoint(other, joint) &&
    Math.abs(other.t - centre.t) > 0 && Math.abs(other.t - centre.t) <= maxGap;
  const prev = samples[index - 1], next = samples[index + 1];
  const hasPrev = usable(prev), hasNext = usable(next);
  const y = centre.jointsWorld[joint][axis];
  const h0 = hasPrev ? centre.t - prev.t : 0;
  const h1 = hasNext ? next.t - centre.t : 0;
  const d0 = hasPrev ? (y - prev.jointsWorld[joint][axis]) / h0 : 0;
  const d1 = hasNext ? (next.jointsWorld[joint][axis] - y) / h1 : 0;
  if (hasPrev && hasNext) {
    if (d0 * d1 <= 0) return 0;
    const w0 = 2 * h1 + h0, w1 = h1 + 2 * h0;
    return (w0 + w1) / (w0 / d0 + w1 / d1);
  }
  // One-sided run endpoints use only observations within that same reliable
  // run. They cannot borrow across a missing or low-confidence neighbour.
  if (hasNext) {
    const next2 = samples[index + 2];
    const h2 = next2 ? next2.t - next.t : 0;
    if (reliableJoint(next2, joint) && h2 > 0 && h2 <= maxGap) {
      const d2 = (next2.jointsWorld[joint][axis] - next.jointsWorld[joint][axis]) / h2;
      return endpointSlope(h1, h2, d1, d2);
    }
    return d1;
  }
  if (hasPrev) {
    const prev2 = samples[index - 2];
    const h2 = prev2 ? prev.t - prev2.t : 0;
    if (reliableJoint(prev2, joint) && h2 > 0 && h2 <= maxGap) {
      const d2 = (prev.jointsWorld[joint][axis] - prev2.jointsWorld[joint][axis]) / h2;
      return endpointSlope(h0, h2, d0, d2);
    }
    return d0;
  }
  return 0;
}

// Appearance frames are transported once along immutable source time. This is
// a coordinate gauge, not a temporal filter of joint positions or playback state.
const rigCache = new WeakMap();
const sub3=(a,b)=>a.map((v,k)=>v-b[k]);
const dot3=(a,b)=>a.reduce((sum,v,k)=>sum+v*b[k],0);
const cross3=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit3=(v,fallback=[0,-1,0])=>{const l=Math.hypot(...v);return l>1e-8?v.map(x=>x/l):fallback.slice();};
const project3=(v,t)=>unit3(v.map((x,k)=>x-dot3(v,t)*t[k]));
function rotate3(v,q){const t=cross3(q.slice(0,3),v).map(x=>2*x),c=cross3(q.slice(0,3),t);return v.map((x,k)=>x+q[3]*t[k]+c[k]);}
function transport3(v,from,to){
 const d=Math.max(-1,Math.min(1,dot3(from,to)));let q;
 if(d<-.999999){const axis=project3(v,from);q=[...axis,0];}
 else{q=[...cross3(from,to),1+d];const n=Math.hypot(...q);q=q.map(x=>x/n);}
 return project3(rotate3(v,q),to);
}
function basisQuaternion(x,y){
 const z=cross3(x,y),m=[x[0],y[0],z[0],x[1],y[1],z[1],x[2],y[2],z[2]],trace=m[0]+m[4]+m[8];let q;
 if(trace>0){const s=Math.sqrt(trace+1)*2;q=[(m[7]-m[5])/s,(m[2]-m[6])/s,(m[3]-m[1])/s,s/4];}
 else if(m[0]>m[4]&&m[0]>m[8]){const s=Math.sqrt(1+m[0]-m[4]-m[8])*2;q=[s/4,(m[1]+m[3])/s,(m[2]+m[6])/s,(m[7]-m[5])/s];}
 else if(m[4]>m[8]){const s=Math.sqrt(1+m[4]-m[0]-m[8])*2;q=[(m[1]+m[3])/s,s/4,(m[5]+m[7])/s,(m[2]-m[6])/s];}
 else{const s=Math.sqrt(1+m[8]-m[0]-m[4])*2;q=[(m[2]+m[6])/s,(m[5]+m[7])/s,s/4,(m[3]-m[1])/s];}
 return q;
}
function slerp4(a,b,u){let dot=a.reduce((s,v,k)=>s+v*b[k],0),end=b;if(dot<0){dot=-dot;end=b.map(v=>-v);}let out;
 if(dot>.9995)out=a.map((v,k)=>v+(end[k]-v)*u);
 else{const theta=Math.acos(Math.min(1,dot)),s=Math.sin(theta);out=a.map((v,k)=>(v*Math.sin((1-u)*theta)+end[k]*Math.sin(u*theta))/s);}
 const n=Math.hypot(...out);return out.map(v=>v/n);
}
function weightedMedian(rows,fallback){if(!rows.length)return fallback;rows.sort((a,b)=>a[0]-b[0]);const half=rows.reduce((s,r)=>s+r[1],0)/2;let sum=0;for(const r of rows){sum+=r[1];if(sum>=half)return r[0];}return rows.at(-1)[0];}
function rigFor(motion,playerId){
 let byPlayer=rigCache.get(motion);if(!byPlayer){byPlayer=new Map();rigCache.set(motion,byPlayer);}if(byPlayer.has(playerId))return byPlayer.get(playerId);
 const player=motion.players[playerId],samples=player.samples,names=motion.jointNames,scale=Math.max(.85,Math.min(1.25,(player.heightMeters??1.88)/1.88));
 const shoulderL=names.indexOf('leftShoulder'),shoulderR=names.indexOf('rightShoulder'),chains={};
 for(const side of ['L','R'])for(const kind of ['arm','leg']){
  const prefix=side==='L'?'left':'right',arm=kind==='arm',indices=[prefix+(arm?'Shoulder':'Hip'),prefix+(arm?'Elbow':'Knee'),prefix+(arm?'Wrist':'Ankle')].map(n=>names.indexOf(n));
  if(indices.some(i=>i<0))continue;const defaults=(arm?[.31,.28]:[.47,.47]).map(v=>v*scale),lengths=[[],[]],frames=[],bendFrames=[];let previous=null,previousBend=null;
  for(const sample of samples){
   for(let b=0;b<2;b++){const i=indices[b],j=indices[b+1],length=Math.hypot(...sub3(sample.jointsWorld[j],sample.jointsWorld[i]));
    const conf=Math.min(sample.confidence[i],sample.confidence[j]);if(conf>=.45&&sample.observed!==false&&length>(arm?.18:.29)*scale&&length<(arm?.50:.72)*scale)lengths[b].push([length,conf*conf]);}
   const axis=unit3(sub3(sample.jointsWorld[indices[1]],sample.jointsWorld[indices[0]]),previous?.axis);
   const right=shoulderL>=0&&shoulderR>=0?unit3(sub3(sample.jointsWorld[shoulderR],sample.jointsWorld[shoulderL]),[1,0,0]):[1,0,0];
   let lateral;
   if(previous)lateral=transport3(previous.lateral,previous.axis,axis);
   else{let reference=right;if(Math.abs(dot3(reference,axis))>.9)reference=unit3(cross3(right,[0,1,0]),[0,0,1]);if(Math.abs(dot3(reference,axis))>.9)reference=[0,1,0];lateral=project3(reference,axis);}
   frames.push(basisQuaternion(lateral,axis));previous={axis,lateral};
   const endAxis=unit3(sub3(sample.jointsWorld[indices[2]],sample.jointsWorld[indices[0]]),previousBend?.axis);
   const rawBend=sub3(sample.jointsWorld[indices[1]],sample.jointsWorld[indices[0]]);
   const projected=rawBend.map((v,k)=>v-dot3(rawBend,endAxis)*endAxis[k]);
   let pole=previousBend?transport3(previousBend.pole,previousBend.axis,endAxis):project3(cross3(lateral,endAxis),endAxis);
   if(Math.hypot(...projected)>.035*scale&&Math.min(...indices.map(i=>sample.confidence[i]))>.45&&sample.observed!==false){
    let candidate=unit3(projected);if(previousBend&&dot3(candidate,pole)<0)candidate=candidate.map(v=>-v);
    pole=project3(candidate,endAxis);
   }
   bendFrames.push(basisQuaternion(unit3(cross3(endAxis,pole)),endAxis));previousBend={axis:endAxis,pole};
  }
  chains[kind+side]={lengths:lengths.map((rows,k)=>weightedMedian(rows,defaults[k])),frames,bendFrames};
 }
 const headings=[],headingConfidence=[];for(const sample of samples){const shoulder=shoulderL>=0&&shoulderR>=0?sub3(sample.jointsWorld[shoulderR],sample.jointsWorld[shoulderL]):[1,0,0];let angle=Math.atan2(-shoulder[2],shoulder[0]);if(headings.length){const previous=headings.at(-1);angle=previous+Math.atan2(Math.sin(angle-previous),Math.cos(angle-previous));}headings.push(angle);
  const confidence=shoulderL>=0&&shoulderR>=0&&sample.observed!==false?Math.min(sample.confidence[shoulderL],sample.confidence[shoulderR]):0;
  headingConfidence.push(confidence*smooth01((Math.hypot(shoulder[0],shoulder[2])-.08*scale)/(.12*scale)));}
 const presentationConfidence=samples.map(sample=>names.map((_,joint)=>{
  let sum=0,weight=0;
  for(const neighbor of samples){const offset=neighbor.t-sample.t;if(Math.abs(offset)>.24)continue;const w=Math.exp(-.5*(offset/.09)**2);sum+=w*(neighbor.observed===false?0:neighbor.confidence[joint]);weight+=w;}
  return sum/(weight||1);
 }));
 const result={chains,headings,headingConfidence,presentationConfidence,frameGauge:'source-time-parallel-transport',lengthBasis:'confidence-weighted-plausible-source-median'};byPlayer.set(playerId,result);return result;
}
function sampleRig(motion,playerId,a,b,lo,hi,u){
 const cached=rigFor(motion,playerId),samples=motion.players[playerId].samples,result={limbLengths:{},limbFrames:{},frameGauge:cached.frameGauge,lengthBasis:cached.lengthBasis};
 const ai=a===samples[lo]?lo:hi,bi=b===samples[hi]?hi:lo;result.headingYaw=cached.headings[ai]+(cached.headings[bi]-cached.headings[ai])*u;result.headingConfidence=cached.headingConfidence[ai]+(cached.headingConfidence[bi]-cached.headingConfidence[ai])*u;
 result.presentationConfidence=cached.presentationConfidence[ai].map((c,i)=>c+(cached.presentationConfidence[bi][i]-c)*u);
 for(const [name,chain]of Object.entries(cached.chains)){const q=slerp4(chain.frames[ai],chain.frames[bi],u),bend=slerp4(chain.bendFrames[ai],chain.bendFrames[bi],u);result.limbLengths[name]=chain.lengths.slice();result.limbFrames[name]={axis:rotate3([0,1,0],q),lateral:rotate3([1,0,0],q),bendAxis:rotate3([0,1,0],bend),bendPole:rotate3([0,0,1],bend)};}
 return result;
}

export function sampleMotion(motion, playerId, t) {
  const player = motion?.players?.[playerId];
  const samples = player?.samples;
  if (!samples?.length || !Number.isFinite(t)) return null;
  const maxGap = motion.maxInterpolationGapSeconds ?? 0.28;
  const fade = motion.edgeFadeSeconds ?? 0.12;
  if (t < samples[0].t - fade || t > samples.at(-1).t + fade) return null;
  let lo = 0, hi = samples.length - 1;
  while (lo + 1 < hi) {
    const mid = (lo + hi) >> 1;
    if (samples[mid].t <= t) lo = mid; else hi = mid;
  }
  let a = samples[lo], b = samples[hi];
  if (t <= samples[0].t) a = b = samples[0];
  if (t >= samples.at(-1).t) a = b = samples.at(-1);
  const dt = b.t - a.t;
  if (dt > maxGap) {
    const near = Math.abs(t-a.t) <= Math.abs(t-b.t) ? a : b;
    const distance = Math.abs(t-near.t);
    if (distance > fade) return null;
    a = b = near;
  }
  const u = a === b ? 0 : Math.min(1, Math.max(0, (t-a.t)/(b.t-a.t)));
  const lerp = (x,y) => x+(y-x)*u;
  const h = b.t - a.t;
  const u2 = u * u, u3 = u2 * u;
  const configuredLimit = motion.maxInterpolationDeviationMeters ?? 0.06;
  const deviationLimit = Number.isFinite(configuredLimit) ? Math.max(0, configuredLimit) : 0.06;
  const jointsWorld = a.jointsWorld.map((j,i) => {
    // Return recorded values directly at the knots, avoiding roundoff from
    // cubic evaluation and preserving exact source-frame comparisons.
    if (u === 0) return j.slice();
    if (u === 1) return b.jointsWorld[i].slice();
    const linear = j.map((v,k) => lerp(v, b.jointsWorld[i][k]));
    if (a === b || !reliableJoint(a, i) || !reliableJoint(b, i)) return linear;
    const curved = j.map((v,k) => {
      const m0 = jointSlope(samples, lo, i, k, maxGap);
      const m1 = jointSlope(samples, hi, i, k, maxGap);
      const end = b.jointsWorld[i][k];
      const value = (2*u3 - 3*u2 + 1)*v + (u3 - 2*u2 + u)*h*m0 +
        (-2*u3 + 3*u2)*end + (u3 - u2)*h*m1;
      return Math.max(Math.min(v, end), Math.min(Math.max(v, end), value));
    });
    // Keep the reconstructed route close to its old linear path. Radial tanh
    // saturation is smooth with derivative 1 at delta=0, so exact knots and
    // reliable-run C1 tangents survive. This remains a convex combination of
    // two coordinate-bounded curves, rather than adding a new extrapolation.
    const delta = curved.map((v,k) => v - linear[k]);
    const distance = Math.hypot(...delta);
    if (distance === 0) return curved;
    const weight = deviationLimit > 0 ? deviationLimit * Math.tanh(distance / deviationLimit) / distance : 0;
    return linear.map((v,k) => v + delta[k] * weight);
  });
  const confidence = a.confidence.map((v,i) => Math.max(0,Math.min(1,lerp(v,b.confidence[i]))));
  const nearestDistance = Math.min(Math.abs(t-a.t),Math.abs(t-b.t));
  const outside = t < samples[0].t || t > samples.at(-1).t;
  const sourceCoverageWeight = outside || a === b ?
    (fade > 0 ? Math.max(0,1-nearestDistance/fade) : Number(nearestDistance === 0)) : 1;
  // A cropped/occluded lower body cannot establish its absolute vertical
  // anchor. Mixing a near-floor inferred pelvis with standing fallback legs
  // creates an impossible rig. Retain true low poses when either ankle is
  // confidently observed, rather than lifting or rejecting all seated poses.
  const pelvisIndex = motion.jointNames.indexOf('pelvis');
  const leftAnkleIndex = motion.jointNames.indexOf('leftAnkle');
  const rightAnkleIndex = motion.jointNames.indexOf('rightAnkle');
  const height = player.heightMeters ?? 1.93;
  let poseSupportWeight = 1, poseFallbackReason;
  if (pelvisIndex >= 0 && leftAnkleIndex >= 0 && rightAnkleIndex >= 0) {
    const anchorHeight = jointsWorld[pelvisIndex][1];
    const ankleConfidence = Math.max(confidence[leftAnkleIndex], confidence[rightAnkleIndex]);
    const verticalSupport = smooth01((anchorHeight / height - .16) / .08);
    const ankleSupport = smooth01((ankleConfidence - .10) / .20);
    poseSupportWeight = 1-(1-verticalSupport)*(1-ankleSupport);
    if (poseSupportWeight < 1) poseFallbackReason = 'low-pelvis-with-unobserved-ground-anchor';
  }
  const trustedFloor= Math.min(...jointsWorld.filter((_,i)=>confidence[i]>=.5).map(j=>j[1]),Infinity);
  const floorSupport=smooth01((trustedFloor+.070)/.055);
  if(floorSupport<1){poseSupportWeight*=floorSupport;poseFallbackReason='confident-joint-below-court';}
  const coverageWeight = sourceCoverageWeight * poseSupportWeight;
  // Left/right shoulders carry an anatomical orientation. The original
  // monocular nose-depth sign is ambiguous and can alternate by 180 degrees
  // while these same shoulders remain still. Do not use raw forwardWorld.
  const left = motion.jointNames.indexOf('leftShoulder');
  const right = motion.jointNames.indexOf('rightShoulder');
  let forwardWorld, orientationConfidence = 0;
  if (left >= 0 && right >= 0 && reliableJoint(a, left) && reliableJoint(a, right) &&
      reliableJoint(b, left) && reliableJoint(b, right)) {
    const dx = jointsWorld[right][0] - jointsWorld[left][0];
    const dz = jointsWorld[right][2] - jointsWorld[left][2];
    const length = Math.hypot(dx, dz);
    const minimumLength = 0.12 * (player.heightMeters ?? 1.93) / 1.93;
    const supported = Math.min(confidence[left], confidence[right]) * coverageWeight;
    if (length >= minimumLength && supported >= SMOOTH_CONFIDENCE) {
      forwardWorld = [-dz / length, 0, dx / length]; // shoulderRight cross up
      orientationConfidence = supported;
    }
  }
  return {jointsWorld, confidence, jointNames:motion.jointNames,
    observed:nearestDistance < 1e-4 && (a.observed !== false || b.observed !== false),
    coverageWeight, sourceCoverageWeight, poseSupportWeight, poseFallbackReason,
    forwardWorld, orientationConfidence,
    facingProvenance:forwardWorld ? 'anatomical-left-right-shoulder-axis' : 'tactical-fallback',
    heightMeters:player.heightMeters,
    time:t,rig:sampleRig(motion,playerId,a,b,lo,hi,u),
    provenance:player.provenance ?? motion.provenance};
}
