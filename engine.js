// Astro Stack — deterministic, browser-local image processing. No network or AI calls.
export const ANALYSIS_SIZE = 96;
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
export function bilinear(a, w, h, x, y, stride = 1, channel = 0) {
  if (x < 0 || y < 0 || x > w - 1 || y > h - 1) return 0;
  const ix = Math.floor(x), iy = Math.floor(y), jx = Math.min(ix + 1, w - 1), jy = Math.min(iy + 1, h - 1), fx = x - ix, fy = y - iy;
  return a[(iy*w+ix)*stride+channel]*(1-fx)*(1-fy)+a[(iy*w+jx)*stride+channel]*fx*(1-fy)+a[(jy*w+ix)*stride+channel]*(1-fx)*fy+a[(jy*w+jx)*stride+channel]*fx*fy;
}
export function blur(a, w, h, sigma = 1) {
  if (sigma <= 0) return new Float32Array(a);
  const r = Math.max(1, Math.ceil(sigma*2.5)), kernel = new Float32Array(r*2+1); let sum=0;
  for(let i=-r;i<=r;i++){const v=Math.exp(-i*i/(2*sigma*sigma));kernel[i+r]=v;sum+=v;}
  for(let i=0;i<kernel.length;i++)kernel[i]/=sum;
  const tmp=new Float32Array(a.length), out=new Float32Array(a.length);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){let v=0;for(let k=-r;k<=r;k++)v+=a[y*w+clamp(x+k,0,w-1)]*kernel[k+r];tmp[y*w+x]=v;}
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){let v=0;for(let k=-r;k<=r;k++)v+=tmp[clamp(y+k,0,h-1)*w+x]*kernel[k+r];out[y*w+x]=v;}
  return out;
}
function gradient(g,n){const gx=new Float32Array(g.length),gy=new Float32Array(g.length);for(let y=1;y<n-1;y++)for(let x=1;x<n-1;x++){const i=y*n+x;gx[i]=(g[i+1]-g[i-1])/2;gy[i]=(g[i+n]-g[i-n])/2;}return {gx,gy};}
function quantile(arr,p){if(!arr.length)return 0;const a=Array.from(arr).sort((a,b)=>a-b);return a[Math.floor((a.length-1)*p)];}
export function detectSubject(rgba,w,h){
  const gray=new Float32Array(w*h);let maximum=0;const border=[];
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){let i=y*w+x;const v=(rgba[i*4]*.25+rgba[i*4+1]*.65+rgba[i*4+2]*.1)/255;gray[i]=v;maximum=Math.max(maximum,v);if(x<3||y<3||x>w-4||y>h-4)border.push(v);}
  const bg=quantile(border,.5),smooth=blur(gray,w,h,.7),threshold=bg+Math.max(.035,(maximum-bg)*.26),seen=new Uint8Array(w*h);let best=null;
  const queue=new Int32Array(w*h);
  for(let p=0;p<gray.length;p++){
    if(seen[p]||smooth[p]<=threshold)continue;
    let head=0,tail=0;queue[tail++]=p;seen[p]=1;let mass=0,cx=0,cy=0,x0=w,x1=0,y0=h,y1=0;
    while(head<tail){const k=queue[head++],y=Math.floor(k/w),x=k%w,v=smooth[k]-bg;mass+=v;cx+=v*x;cy+=v*y;x0=Math.min(x0,x);x1=Math.max(x1,x);y0=Math.min(y0,y);y1=Math.max(y1,y);for(const q of [x>0?k-1:-1,x<w-1?k+1:-1,y>0?k-w:-1,y<h-1?k+w:-1])if(q>=0&&!seen[q]&&smooth[q]>threshold){seen[q]=1;queue[tail++]=q;}}
    if(tail>=3&&(!best||mass>best.mass))best={mass,cx:cx/mass,cy:cy/mass,x:x0,y:y0,w:x1-x0+1,h:y1-y0+1,area:tail};
  }
  return best?{...best,found:true,bg,max:maximum}:{found:false,cx:w/2,cy:h/2,x:0,y:0,w,h,bg,max:maximum};
}
export function analyzeFrame(rgba,n){
  const s=ANALYSIS_SIZE,gray=new Float32Array(s*s);let saturated=0;
  for(let y=0;y<s;y++)for(let x=0;x<s;x++){const xx=(x+.5)*n/s-.5,yy=(y+.5)*n/s-.5;gray[y*s+x]=(bilinear(rgba,n,n,xx,yy,4,0)*.25+bilinear(rgba,n,n,xx,yy,4,1)*.65+bilinear(rgba,n,n,xx,yy,4,2)*.1)/255;}
  const edge=[];for(let y=0;y<s;y++)for(let x=0;x<s;x++)if(x<8||y<8||x>s-9||y>s-9)edge.push(gray[y*s+x]);
  const bg=quantile(edge,.5);let peak=0;
  for(let i=0;i<gray.length;i++){gray[i]=Math.max(0,gray[i]-bg);peak=Math.max(peak,gray[i]);if(gray[i]>.98)saturated++;}
  const smooth=blur(gray,s,s,.7),{gx,gy}=gradient(smooth,s);let mass=0,cx=0,cy=0,energy=0,lapEnergy=0,area=0;
  for(let y=1;y<s-1;y++)for(let x=1;x<s-1;x++){const i=y*s+x,v=Math.max(0,smooth[i]-peak*.12);mass+=v;cx+=x*v;cy+=y*v;if(smooth[i]>peak*.1){const lap=smooth[i-1]+smooth[i+1]+smooth[i-s]+smooth[i+s]-4*smooth[i];lapEnergy+=lap*lap;energy+=smooth[i]*smooth[i];area++;}}
  cx=mass?cx/mass:s/2;cy=mass?cy/mass:s/2;let xx=0,yy=0,xy=0;
  for(let y=0;y<s;y++)for(let x=0;x<s;x++){const v=Math.max(0,smooth[y*s+x]-peak*.12);xx+=v*(x-cx)**2;yy+=v*(y-cy)**2;xy+=v*(x-cx)*(y-cy);}
  xx/=mass||1;yy/=mass||1;xy/=mass||1;const discriminant=Math.sqrt((xx-yy)**2+4*xy*xy),major=(xx+yy+discriminant)/2,minor=Math.max(.01,(xx+yy-discriminant)/2),angle=.5*Math.atan2(2*xy,xx-yy),aspect=Math.sqrt(major/minor),radius=Math.sqrt((major+minor)/2);
  let noiseSum=0,noiseCount=0;for(let y=2;y<s-2;y++)for(let x=2;x<s-2;x++){const i=y*s+x;if(Math.hypot(x-s/2,y-s/2)>s*.43&&smooth[i]<peak*.12){const lap=smooth[i-1]+smooth[i+1]+smooth[i-s]+smooth[i+s]-4*smooth[i];noiseSum+=lap*lap;noiseCount++;}}const noise=noiseSum/Math.max(1,noiseCount),detail=Math.max(0,lapEnergy-noise*area);const quality=detail/Math.max(energy,1e-9)/(1+6*noise*area/Math.max(detail,.00001))*(1-Math.min(.8,saturated/(area||1)));
  return {gray:smooth,gx,gy,cx,cy,angle,aspect,radius,quality,saturated:saturated/(area||1),peak,signal:area,valid:mass>.05&&peak>.025};
}
function wrapAngle(x){while(x>Math.PI/2)x-=Math.PI;while(x< -Math.PI/2)x+=Math.PI;return x;}
function initialPose(src,ref){
  const theta=ref.aspect>1.2&&src.aspect>1.2?wrapAngle(ref.angle-src.angle):0,scale=clamp(ref.radius/Math.max(src.radius,1),.72,1.4),c=Math.cos(theta),sn=Math.sin(theta),C=ANALYSIS_SIZE/2;
  return {tx:ref.cx-C-scale*(c*(src.cx-C)-sn*(src.cy-C)),ty:ref.cy-C-scale*(sn*(src.cx-C)+c*(src.cy-C)),theta,scale};
}
function referenceSamples(ref){
  const n=ANALYSIS_SIZE,C=n/2,ax=Math.cos(ref.angle),ay=Math.sin(ref.angle),planet=ref.aspect>1.2;let points=[],energies=[0,0,0];
  for(let y=5;y<n-5;y+=2)for(let x=5;x<n-5;x+=2){const i=y*n+x,dx=x-ref.cx,dy=y-ref.cy,u=ax*dx+ay*dy,v=-ay*dx+ax*dy;let region=0;
    if(planet){const r=Math.max(8,ref.radius*1.05),tip=Math.max(12,ref.radius*1.15);if(dx*dx+dy*dy<r*r)region=0;else if(u>tip&&Math.abs(v)<ref.radius*.9)region=1;else if(u< -tip&&Math.abs(v)<ref.radius*.9)region=2;else continue;}
    else{if(ref.gray[i]<ref.peak*.05)continue;region=x<C-12?1:x>C+12?2:0;}
    if(ref.gray[i]<ref.peak*.025)continue;points.push({x,y,i,region,t:ref.gray[i],gx:ref.gx[i],gy:ref.gy[i]});energies[region]+=ref.gray[i]**2;
  }
  for(const p of points)p.w=(p.region===0?.46:.27)/Math.max(energies[p.region],.0001);
  return points;
}
function lossPose(src,ref,points,p,normalize=true,regions=false){
  const n=ANALYSIS_SIZE,C=n/2,c=Math.cos(p.theta),sn=Math.sin(p.theta),inv=1/p.scale;let dot=0,norm=0;const values=[];
  for(const z of points){const dx=z.x-C-p.tx,dy=z.y-C-p.ty,x=(c*dx+sn*dy)*inv+C,y=(-sn*dx+c*dy)*inv+C,v=bilinear(src.gray,n,n,x,y);dot+=v*z.t*z.w;norm+=v*v*z.w;values.push([x,y,v]);}
  const gain=normalize?clamp(dot/Math.max(norm,1e-12),.4,2.5):1;let loss=0;const err=[0,0,0],den=[0,0,0];
  for(let j=0;j<points.length;j++){const z=points[j],[x,y,v]=values[j],a=bilinear(src.gx,n,n,x,y),b=bilinear(src.gy,n,n,x,y),gx=(c*a-sn*b)*inv*gain,gy=(sn*a+c*b)*inv*gain,d=v*gain-z.t;loss+=z.w*(d*d+1.4*((gx-z.gx)**2+(gy-z.gy)**2));err[z.region]+=d*d;den[z.region]+=z.t*z.t;}
  return regions?{loss,gain,regions:err.map((v,i)=>Math.sqrt(v/Math.max(den[i],1e-9)))}:loss;
}
export function fitFrame(src,ref,{normalize=true,precision='fine'}={}){
  const points=referenceSamples(ref);let p=initialPose(src,ref),best=lossPose(src,ref,points,p,normalize);const initial={...p};
  const levels=precision==='fast'?[2,1,.5,.25]:[2,1,.5,.25,.125,.0625,.03125];
  for(const step of levels){for(let cycle=0;cycle<3;cycle++){let improved=false;for(const key of ['tx','ty','theta','scale']){const delta=key==='theta'?step*Math.PI/180:key==='scale'?step*.012:step;let winner=null;for(const direction of [-1,1]){const q={...p,[key]:p[key]+direction*delta};if(q.scale<.65||q.scale>1.5||Math.abs(q.tx-initial.tx)>14||Math.abs(q.ty-initial.ty)>14||Math.abs(q.theta-initial.theta)>.25)continue;const value=lossPose(src,ref,points,q,normalize);if(value<best-1e-8){best=value;winner=q;}}if(winner){p=winner;improved=true;}}if(!improved)break;}}
  return {...p,...lossPose(src,ref,points,p,normalize,true)};
}
export function toOutputPose(p,n){return {dx:p.tx*n/ANALYSIS_SIZE,dy:p.ty*n/ANALYSIS_SIZE,rotation:p.theta*180/Math.PI,scale:p.scale,gain:p.gain,error:p.loss,regions:p.regions};}
export function effectivePose(transform,manual={}){return {...transform,dx:transform.dx+(+manual.dx||0),dy:transform.dy+(+manual.dy||0),rotation:transform.rotation+(+manual.rotation||0),scale:transform.scale*(1+(+manual.scale||0)/100)};}
export function warpRGBA(rgba,inputN,transform,outputN){
  const out=new Uint8ClampedArray(outputN*outputN*4),p=transform,C=outputN/2,c=Math.cos(p.rotation*Math.PI/180),sn=Math.sin(p.rotation*Math.PI/180),ratio=inputN/outputN,inv=1/p.scale,channels=p.channelShifts||{};
  for(let y=0;y<outputN;y++)for(let x=0;x<outputN;x++){const dx=x-C-p.dx,dy=y-C-p.dy,sx=((c*dx+sn*dy)*inv+C)*ratio,sy=((-sn*dx+c*dy)*inv+C)*ratio,i=(y*outputN+x)*4;
    for(let ch=0;ch<3;ch++){const d=channels[ch]||[0,0];out[i+ch]=bilinear(rgba,inputN,inputN,sx-d[0],sy-d[1],4,ch)*(p.gain||1);}out[i+3]=255;
  }return out;
}
export function chooseReference(frames,analyses){
  const valid=frames.map((f,i)=>({f,a:analyses[i]})).filter(v=>v.a.valid).sort((x,y)=>y.a.quality-x.a.quality);if(!valid.length)throw Error('未检测到足够的主体信号，请重新圈定范围或换用更明亮的素材。');
  const top=valid.slice(0,Math.max(1,Math.ceil(valid.length*.35))),radii=top.map(x=>x.a.radius),median=quantile(radii,.5);top.sort((x,y)=>Math.abs(x.a.radius-median)-Math.abs(y.a.radius-median));return top[0].f.id;
}
export function estimateChannelShifts(rgba,n){
  const s=48,channels=[];for(let ch=0;ch<3;ch++){const g=new Float32Array(s*s);for(let y=0;y<s;y++)for(let x=0;x<s;x++)g[y*s+x]=bilinear(rgba,n,n,(x+.5)*n/s-.5,(y+.5)*n/s-.5,4,ch)/255;channels.push(blur(g,s,s,.8));}
  const ref=channels[1],out={};
  for(const ch of [0,2]){const a=channels[ch];let best={x:0,y:0},score=Infinity;
    const loss=(dx,dy)=>{let dot=0,norm=0,energy=0;for(let y=6;y<s-6;y+=2)for(let x=6;x<s-6;x+=2){const t=ref[y*s+x],v=bilinear(a,s,s,x-dx,y-dy);dot+=t*v;norm+=v*v;energy+=t*t;}return 1-dot*dot/Math.max(norm*energy,1e-12);};
    score=loss(0,0);for(const step of [.5,.25,.125])for(let cycle=0;cycle<3;cycle++){let change=false;for(const axis of ['x','y'])for(const dir of [-1,1]){const q={...best,[axis]:best[axis]+dir*step};if(Math.abs(q.x)>1.5||Math.abs(q.y)>1.5)continue;const v=loss(q.x,q.y);if(v<score-1e-7){score=v;best=q;change=true;}}if(!change)break;}
    out[ch]=[best.x*n/s,best.y*n/s];
  }return out;
}
export async function stackFrames(frames,transforms,n,{method='weighted',normalize=true,onProgress=()=>{},cancelled=()=>false}={}){
  if(!frames.length)throw Error('请至少选择一帧。');const out=new Float32Array(n*n*3);let weightSum=0;const qs=frames.map(f=>f.quality||.001),median=quantile(qs,.5)||.001;
  for(let k=0;k<frames.length;k++){
    if(cancelled())throw Error('操作已取消。');const f=frames[k],p=transforms[f.id];if(!p)throw Error('存在尚未对齐的帧，请先完成自动对齐。');
    const C=n/2,c=Math.cos(p.rotation*Math.PI/180),sn=Math.sin(p.rotation*Math.PI/180),ratio=f.n/n,inv=1/p.scale,weight=method==='weighted'?clamp((f.quality||median)/median,.4,2):1,gain=normalize?p.gain||1:1,channels=p.channelShifts||{};weightSum+=weight;
    for(let y=0;y<n;y++)for(let x=0;x<n;x++){const dx=x-C-p.dx,dy=y-C-p.dy,sx=((c*dx+sn*dy)*inv+C)*ratio,sy=((-sn*dx+c*dy)*inv+C)*ratio,i=(y*n+x)*3;for(let ch=0;ch<3;ch++){const d=channels[ch]||[0,0];out[i+ch]+=bilinear(f.rgba,f.n,f.n,sx-d[0],sy-d[1],4,ch)/255*weight*gain;}}
    onProgress(k+1,frames.length);if(k%2===0)await new Promise(r=>setTimeout(r,0));
  }
  for(let i=0;i<out.length;i++)out[i]/=weightSum;return out;
}
export function editStack(raw,n,settings={}){
  const {exposure=0,contrast=0,gamma=1,saturation=1,temperature=0,sharpness=0,denoise=0,blackpoint=0}=settings;const planes=[new Float32Array(n*n),new Float32Array(n*n),new Float32Array(n*n)];
  for(let i=0;i<n*n;i++)for(let ch=0;ch<3;ch++)planes[ch][i]=raw[i*3+ch];
  if(denoise>0)for(let ch=0;ch<3;ch++){const sm=blur(planes[ch],n,n,.8);for(let i=0;i<sm.length;i++)planes[ch][i]=planes[ch][i]*(1-denoise*.6)+sm[i]*denoise*.6;}
  if(sharpness>0)for(let ch=0;ch<3;ch++){const sm=blur(planes[ch],n,n,Math.max(.7,n/400));for(let i=0;i<sm.length;i++)planes[ch][i]+=sharpness*(planes[ch][i]-sm[i]);}
  const out=new Uint8ClampedArray(n*n*4),boost=2**exposure,c=1+contrast;
  for(let i=0;i<n*n;i++){let r=Math.max(0,planes[0][i]-blackpoint)*boost*(1+temperature*.3),g=Math.max(0,planes[1][i]-blackpoint)*boost,b=Math.max(0,planes[2][i]-blackpoint)*boost*(1-temperature*.3),l=.2126*r+.7152*g+.0722*b;r=l+(r-l)*saturation;g=l+(g-l)*saturation;b=l+(b-l)*saturation;for(const [ch,v] of [[0,r],[1,g],[2,b]])out[i*4+ch]=255*clamp(((v-.18)*c+.18),0,1)**(1/Math.max(.1,gamma));out[i*4+3]=255;}
  return out;
}
export function encodeTiff16(raw,n){
  const entries=10,extras=8+2+entries*12+4,pixelOffset=extras+6,buffer=new ArrayBuffer(pixelOffset+n*n*6),v=new DataView(buffer);v.setUint8(0,73);v.setUint8(1,73);v.setUint16(2,42,true);v.setUint32(4,8,true);v.setUint16(8,entries,true);
  const tags=[[256,4,1,n],[257,4,1,n],[258,3,3,extras],[259,3,1,1],[262,3,1,2],[273,4,1,pixelOffset],[277,3,1,3],[278,4,1,n],[279,4,1,n*n*6],[284,3,1,1]];
  tags.forEach((t,i)=>{const o=10+i*12;v.setUint16(o,t[0],true);v.setUint16(o+2,t[1],true);v.setUint32(o+4,t[2],true);v.setUint32(o+8,t[3],true);});v.setUint16(extras,16,true);v.setUint16(extras+2,16,true);v.setUint16(extras+4,16,true);for(let i=0;i<raw.length;i++)v.setUint16(pixelOffset+i*2,Math.round(clamp(raw[i],0,1)*65535),true);return buffer;
}

export function measureAlignedDifference(current, reference, n){
 const src=analyzeFrame(current,n),ref=analyzeFrame(reference,n),points=referenceSamples(ref);
 return lossPose(src,ref,points,{tx:0,ty:0,theta:0,scale:1},true,true).regions;
}
