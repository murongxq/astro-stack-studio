import {analyzeFrame,chooseReference,fitFrame,toOutputPose,estimateChannelShifts,stackFrames,editStack} from './engine.js';
let cancel=false;
self.onmessage=async({data:m})=>{
  if(m.type==='cancel'){cancel=true;return;}cancel=false;
  try{
    if(m.type==='align'){
      const {frames,n,options}=m,analyses=frames.map(f=>analyzeFrame(f.rgba,f.n));let reference=options.referenceId||(frames.length===1?frames[0].id:chooseReference(frames,analyses));const ri=frames.findIndex(f=>f.id===reference);if(ri<0)throw Error('参考帧已被取消选择，请重新设定。');const transforms={};
      for(let i=0;i<frames.length;i++){
        if(cancel)throw Error('操作已取消。');const f=frames[i],fit=i===ri?{tx:0,ty:0,theta:0,scale:1,gain:1,loss:0,regions:[0,0,0]}:fitFrame(analyses[i],analyses[ri],options);const p=toOutputPose(fit,n);if(options.channels)p.channelShifts=estimateChannelShifts(f.rgba,f.n);transforms[f.id]=p;
        self.postMessage({type:'progress',phase:'align',current:i+1,total:frames.length});await new Promise(r=>setTimeout(r,0));
      }
      self.postMessage({type:'aligned',transforms,reference,roundSubject:analyses[ri].aspect<1.2});
    }else if(m.type==='stack'){
      const raw=await stackFrames(m.frames,m.transforms,m.n,{...m.options,cancelled:()=>cancel,onProgress:(current,total)=>self.postMessage({type:'progress',phase:'stack',current,total})});self.postMessage({type:'stacked',raw,n:m.n},[raw.buffer]);
    }else if(m.type==='edit'){
      const rgba=editStack(m.raw,m.n,m.settings);self.postMessage({type:'edited',rgba,requestId:m.requestId},[rgba.buffer]);
    }
  }catch(error){self.postMessage({type:'error',message:error.message||String(error)});}
};
