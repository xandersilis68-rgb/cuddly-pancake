// Volleyball Analyzer with skill detection + event markers
let detector;
const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const videoInput = document.getElementById('videoInput');
const startBtn = document.getElementById('startBtn');
const statusEl = document.getElementById('status');
const metricsEl = document.getElementById('metrics');
const tipsEl = document.getElementById('tips');
const downloadCsvBtn = document.getElementById('downloadCsvBtn');
const skillSelect = document.getElementById('skill');

let framesData = [];
let eventMarkers = {}; // takeoff, peak, contact, toss

async function loadModel() {
  statusEl.textContent = 'Loading model...';
  await tf.ready();
  const model = poseDetection.SupportedModels.MoveNet;
  const detectorConfig = {modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING};
  detector = await poseDetection.createDetector(model, detectorConfig);
  statusEl.textContent = 'Model loaded. Ready.';
  startBtn.disabled = false;
}
videoInput.addEventListener('change', ev => {
  const file = ev.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  video.src = url;
  video.load();
  video.addEventListener('loadedmetadata', () => {
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
  }, {once:true});
  startBtn.disabled = false;
});
startBtn.addEventListener('click', async () => {
  if (!detector) { statusEl.textContent = 'Model not ready'; return; }
  if (!video.src) { statusEl.textContent = 'Please pick a video.'; return; }
  framesData = []; eventMarkers = {};
  tipsEl.innerHTML = ''; metricsEl.innerHTML = '';
  downloadCsvBtn.disabled = true;
  video.currentTime = 0; video.muted = true;
  await video.play();
  statusEl.textContent = 'Analyzing... (playing)';
  analyzeLoop();
});

async function analyzeLoop() {
  const fpsSample = 12;
  const intervalMs = 1000 / fpsSample;
  let last = performance.now();

  async function processFrame() {
    if (video.paused || video.ended) { finishAnalysis(); return; }
    const now = performance.now();
    if (now - last < intervalMs) { requestAnimationFrame(processFrame); return; }
    last = now;

    ctx.clearRect(0,0,overlay.width,overlay.height);
    ctx.drawImage(video,0,0,overlay.width,overlay.height);

    const poses = await detector.estimatePoses(video,{flipHorizontal:false});
    if (poses && poses.length) {
      const pose = poses[0];
      drawKeypointsAndSkeleton(pose.keypoints);
      const m = computeMetrics(pose.keypoints);
      framesData.push({time: video.currentTime, ...m});
      updateUIAggregates(framesData);
      drawEventMarkers();
    }
    requestAnimationFrame(processFrame);
  }
  requestAnimationFrame(processFrame);
}

// drawing keypoints
function drawKeypointsAndSkeleton(kps) {
  for (const kp of kps) {
    if (kp.score > 0.3) {
      ctx.beginPath();
      ctx.arc(kp.x, kp.y, 4, 0, Math.PI*2);
      ctx.fillStyle = 'rgba(255, 0, 0, 0.9)';
      ctx.fill();
    }
  }
  const edges = [
    ['left_shoulder','right_shoulder'],['left_shoulder','left_elbow'],
    ['left_elbow','left_wrist'],['right_shoulder','right_elbow'],
    ['right_elbow','right_wrist'],['left_hip','left_knee'],
    ['left_knee','left_ankle'],['right_hip','right_knee'],
    ['right_knee','right_ankle'],['left_shoulder','left_hip'],
    ['right_shoulder','right_hip']
  ];
  const nameToKp = Object.fromEntries(kps.map(k=>[k.name,k]));
  ctx.lineWidth=2; ctx.strokeStyle='rgba(0,255,200,0.9)';
  for (const [a,b] of edges) {
    const A=nameToKp[a],B=nameToKp[b];
    if (A?.score>0.3 && B?.score>0.3) {
      ctx.beginPath(); ctx.moveTo(A.x,A.y); ctx.lineTo(B.x,B.y); ctx.stroke();
    }
  }
}
function dist(a,b){return Math.hypot(a.x-b.x,a.y-b.y);}

function computeMetrics(kps) {
  const kp=Object.fromEntries(kps.map(k=>[k.name,k]));
  const hipMid={x:(kp.left_hip.x+kp.right_hip.x)/2,y:(kp.left_hip.y+kp.right_hip.y)/2};
  const shoulderMid={x:(kp.left_shoulder.x+kp.right_shoulder.x)/2,y:(kp.left_shoulder.y+kp.right_shoulder.y)/2};
  const ankleMid={x:(kp.left_ankle.x+kp.right_ankle.x)/2,y:(kp.left_ankle.y+kp.right_ankle.y)/2};
  const torsoLen=dist(shoulderMid,ankleMid);
  const leftArm=(kp.left_wrist.score>0.3&&kp.left_shoulder.score>0.3)?dist(kp.left_wrist,kp.left_shoulder):0;
  const rightArm=(kp.right_wrist.score>0.3&&kp.right_shoulder.score>0.3)?dist(kp.right_wrist,kp.right_shoulder):0;
  const armExt=Math.max(leftArm,rightArm);
  const shoulderSpread=Math.abs(kp.left_shoulder.x-kp.right_shoulder.x);
  const hipSpread=Math.abs(kp.left_hip.x-kp.right_hip.x);
  const shoulderHipRatio=hipSpread>0?(shoulderSpread/hipSpread):1;
  const leftKneeBend=(kp.left_knee.score>0.3&&kp.left_hip.score>0.3&&kp.left_ankle.score>0.3)?
    ((kp.left_knee.y-kp.left_hip.y)/(kp.left_ankle.y-kp.left_hip.y)):null;
  const rightKneeBend=(kp.right_knee.score>0.3&&kp.right_hip.score>0.3&&kp.right_ankle.score>0.3)?
    ((kp.right_knee.y-kp.right_hip.y)/(kp.right_ankle.y-kp.right_hip.y)):null;
  return {time:video.currentTime,hipY:hipMid.y,torsoLen,armExt,shoulderHipRatio,leftKneeBend,rightKneeBend};
}

// skill detection
function detectSkill(summary){
  const {jumpRelative,armExtRel,shoulderAvg,kneeMedian}=summary;
  let serve=0,spike=0,set=0;
  if(jumpRelative<0.18&&armExtRel>0.85)serve+=2;
  if(shoulderAvg&&shoulderAvg>1.05)serve+=1;
  if(kneeMedian&&kneeMedian>0.4&&kneeMedian<0.6)serve+=1;
  if(jumpRelative>=0.18)spike+=2;
  if(armExtRel>0.9)spike+=2;
  if(shoulderAvg&&shoulderAvg>1.05)spike+=1;
  if(armExtRel<0.85)set+=2;
  if(jumpRelative<0.15)set+=1;
  if(kneeMedian&&kneeMedian>0.5)set+=1;
  const scores={serve,spike,set};
  const best=Object.entries(scores).sort((a,b)=>b[1]-a[1])[0];
  if(best[1]===0)return'general';return best[0];
}

function updateUIAggregates(frames){
  if(!frames.length)return;
  const baselineFrames=frames.filter(f=>f.time<1.0);
  const baseline=baselineFrames.length?median(baselineFrames.map(f=>f.hipY)):frames[0].hipY;
  const hipYs=frames.map(f=>f.hipY);
  const minHip=Math.min(...hipYs);
  const jumpPx=Math.max(0,baseline-minHip);
  const torsoMedian=median(frames.map(f=>f.torsoLen).filter(Boolean))||1;
  const jumpRelative=jumpPx/torsoMedian;
  const armMax=Math.max(...frames.map(f=>f.armExt));
  const armExtRel=armMax/torsoMedian;
  const shoulderAvg=median(frames.map(f=>f.shoulderHipRatio).filter(Boolean));
  const knees=frames.flatMap(f=>[f.leftKneeBend,f.rightKneeBend].filter(Boolean));
  const kneeMedian=knees.length?median(knees):null;

  let skill=skillSelect.value;
  if(document.getElementById('autoDetect').checked){
    skill=detectSkill({jumpRelative,armExtRel,shoulderAvg,kneeMedian});
    skillSelect.value=skill;
  }

  // event markers
  if(!eventMarkers.takeoff && jumpRelative>0.15)eventMarkers.takeoff=frames.find(f=>f.hipY===Math.max(...hipYs));
  eventMarkers.peak=frames.find(f=>f.hipY===minHip);
  eventMarkers.contact=frames.find(f=>f.armExt===armMax);
  if(skill==='serve'&&!eventMarkers.toss){
    const wristYs=frames.map(f=>f.armExt);
    const rising=frames.find(f=>f.armExt===Math.min(...wristYs));
    eventMarkers.toss=rising;
  }

  const tips=[];
  if(skill==='general'){ if(jumpRelative<0.15)tips.push('Jump seems low.'); else if(jumpRelative>0.3)tips.push('Good vertical!'); if(armExtRel<0.9)tips.push('Extend arm fully.'); else tips.push('Good arm extension.');}
  if(skill==='serve'){ if(armExtRel<0.95)tips.push('Extend hitting arm more.'); if(kneeMedian&&kneeMedian<0.4)tips.push('Add slight knee bend.'); tips.push('Work on toss consistency.');}
  if(skill==='spike'){ if(jumpRelative<0.2)tips.push('Increase vertical.'); if(armExtRel<1.0)tips.push('Reach higher at contact.'); tips.push('Snap wrist over the ball.');}
  if(skill==='set'){ if(armExtRel>0.8)tips.push('Keep elbows bent slightly.'); if(kneeMedian&&kneeMedian<0.5)tips.push('Use more knee bend.'); tips.push('Ensure both hands contact ball evenly.');}

  metricsEl.innerHTML='';
  addMetric('Skill',skill);
  addMetric('Jump (rel)',jumpRelative.toFixed(2));
  addMetric('Arm reach',armExtRel.toFixed(2));
  addMetric('Shoulder/Hip', (shoulderAvg||0).toFixed(2));
  addMetric('Knee bend', kneeMedian? kneeMedian.toFixed(2):'n/a');
  tipsEl.innerHTML='';
  for(const t of tips){const li=document.createElement('li');li.textContent=t;tipsEl.appendChild(li);}
  if(frames.length>5)downloadCsvBtn.disabled=false;
  window.latestAnalysisSummary={skill,jumpRelative,armExtRel,shoulderAvg,kneeMedian,tips};
}
function addMetric(name,val){const d=document.createElement('div');d.className='metric';d.innerHTML=`<strong>${name}</strong><div>${val}</div>`;metricsEl.appendChild(d);}
function median(arr){const s=arr.slice().sort((a,b)=>a-b);if(!s.length)return 0;const m=Math.floor(s.length/2);return s.length%2? s[m]:(s[m-1]+s[m])/2;}

// draw markers
function drawEventMarkers(){
  ctx.font='16px Arial';ctx.fillStyle='yellow';
  const draw=(ev,label)=>{if(ev){ctx.fillText(label,20,20+Object.keys(eventMarkers).indexOf(label)*20);}};
  if(eventMarkers.takeoff)draw(eventMarkers.takeoff,'Takeoff');
  if(eventMarkers.peak)draw(eventMarkers.peak,'Peak');
  if(eventMarkers.contact)draw(eventMarkers.contact,'Contact');
  if(eventMarkers.toss)draw(eventMarkers.toss,'Toss');
}

function finishAnalysis(){
  statusEl.textContent='Analysis finished.';video.pause();
  downloadCsvBtn.onclick=()=>{
    const rows=[['time','hipY','torsoLen','armExt','shoulderHipRatio','leftKneeBend','rightKneeBend']];
    for(const r of framesData){rows.push([r.time.toFixed(3),r.hipY,r.torsoLen,r.armExt,r.shoulderHipRatio,r.leftKneeBend,r.rightKneeBend]);}
    const summary=window.latestAnalysisSummary||{};
    rows.push([]);rows.push(['Summary','Skill (detected)',summary.skill]);rows.push(['Summary','Jump',summary.jumpRelative]);rows.push(['Summary','Arm',summary.armExtRel]);rows.push(['Summary','Shoulder/Hip',summary.shoulderAvg]);rows.push(['Summary','Knee',summary.kneeMedian]);for(const t of summary.tips||[])rows.push(['Tip',t]);
    const csv=rows.map(r=>r.join(',')).join('\n');const blob=new Blob([csv],{type:'text/csv'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='volleyball_analysis.csv';a.click();};
}

loadModel();
