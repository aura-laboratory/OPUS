// Regression checks for tab visibility, map layout ordering and evaluation feedback.
// Uses Node's built-in test runner; no browser or OPUS server is started.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const root=path.join(__dirname,'..','opus_rl','app','frontend');

function harness(){
  const html=fs.readFileSync(path.join(root,'index.html'),'utf8'),nodes=new Map(),tabNodes=[],nav=[];
  function node(id=''){
    return {id,hidden:false,dataset:{},style:{},textContent:'',innerHTML:'',value:'',disabled:false,
      classList:{toggle(){}},setAttribute(){},removeAttribute(){},replaceChildren(){this.innerHTML='';},
      querySelector(){return null;},querySelectorAll(){return [];}};
  }
  for(const tag of html.matchAll(/<[^!\/][^>]*>/g)){
    const id=tag[0].match(/\bid="([^"]+)"/)?.[1],tabs=tag[0].match(/\bdata-tabs="([^"]+)"/)?.[1];
    if(id||tabs){const element=node(id);element.hidden=/\shidden(?:\s|>)/.test(tag[0]);if(id)nodes.set(id,element);if(tabs){element.dataset.tabs=tabs;tabNodes.push(element);}}
    const tab=tag[0].match(/\bdata-tab="([^"]+)"/)?.[1];if(tab){const element=node();element.dataset.tab=tab;nav.push(element);}
  }
  nodes.get('simulation-policy').value='heuristic';
  const modelOption={textContent:'Model',hidden:true,disabled:true};
  nodes.get('simulation-policy').querySelector=()=>modelOption;
  const frames=[],fits=[],polylines=[];let zoom=1,contained=false,invalidated=false;
  const fakeMap={invalidateSize(){invalidated=true;},getBoundsZoom(){return 18.35;},getBounds(){return {contains:()=>contained};},
    getZoom:()=>zoom,fitBounds(bounds,options){assert.ok(invalidated,'fit must follow map layout invalidation');fits.push({bounds,options});contained=true;zoom=18.35;},getContainer:()=>({style:{}})};
  const context=vm.createContext({document:{getElementById:id=>nodes.get(id),querySelectorAll:selector=>selector==='[data-tabs]'?tabNodes:selector==='nav button'?nav:[]},
    window:{devicePixelRatio:1},performance:{now:()=>0},requestAnimationFrame:callback=>{frames.push(callback);return frames.length;},
    setTimeout:()=>0,clearTimeout(){},fetch:async()=>({ok:true,json:async()=>[]}),structuredClone,
    L:{latLngBounds:points=>({points}),point:(x,y)=>({x,y}),polyline:(points,options)=>{polylines.push({points,options});return {addTo(){return this;},bindTooltip(){return this;}}}},
    fakeMap,testFrame:{polygon_wgs84:[[22.94,40.63],[22.941,40.63],[22.941,40.631]]}});
  vm.runInContext(fs.readFileSync(path.join(root,'field-help.js'),'utf8'),context);
  let source=fs.readFileSync(path.join(root,'app.js'),'utf8');source=source.slice(0,source.lastIndexOf('\ninitializeFieldHelp();'));
  vm.runInContext(source,context);
  vm.runInContext('map=fakeMap;lastFrame=testFrame;',context);
  return {nodes,context,fits,polylines,modelOption,run:code=>vm.runInContext(code,context),flush(){while(frames.length)frames.shift()();}};
}

test('Durations retain seconds and show hours, minutes and seconds across boundaries',()=>{
  const h=harness();
  for(const [seconds,clock] of [[0,'0 h 0 min 0 s'],[59,'0 h 0 min 59 s'],[60,'0 h 1 min 0 s'],[3600,'1 h 0 min 0 s'],[3725,'1 h 2 min 5 s'],[90061,'25 h 1 min 1 s']]){
    assert.equal(h.run(`formatDuration(${seconds})`),`${seconds.toLocaleString()} s (${clock})`);
  }
  assert.equal(h.run('formatDuration(60.5,1)'),'60.5 s (0 h 1 min 0 s)');
});

test('Mission shows setup only; Test owns playback and Training retains metrics',async()=>{
  const h=harness();
  h.run('setTab("mission")');
  for(const id of ['simulation-bar','simulation-controls','live-metrics','safety-metrics','fleet-status','evaluation-panel'])assert.equal(h.nodes.get(id).hidden,true,id);
  h.run('setTab("training")');
  assert.equal(h.nodes.get('simulation-controls').hidden,true);
  assert.equal(h.nodes.get('live-metrics').hidden,false);
  assert.equal(h.nodes.get('charts').hidden,false);
  h.run('setTab("test")');
  assert.equal(h.nodes.get('simulation-controls').hidden,false);
  assert.equal(h.nodes.get('live-metrics').hidden,false);
  assert.equal(h.nodes.get('charts').hidden,false);
  await Promise.resolve();
});

test('Replay framing waits for layout and uses small equal padding',()=>{
  const h=harness();h.run('scheduleMapFit(true)');
  assert.equal(h.fits.length,0,'must not fit before the animation-frame layout callback');
  h.flush();assert.equal(h.fits.length,1);
  assert.deepEqual(Array.from(h.fits[0].options.padding),[12,12]);
  assert.equal(h.fits[0].options.animate,false);
  h.run('scheduleMapFit()');h.flush();assert.equal(h.fits.length,1,'already close: keep the view');
  h.run('setTab("test")');h.flush();assert.equal(h.fits.length,2,'tab changes explicitly reframe');
});

test('Test and Training render UAV paths while Mission hides them',()=>{
  const h=harness();let clears=0;h.context.trailLayer={clearLayers(){clears++;}};
  h.nodes.get('test-trail-length').value='2';
  h.run('trailGroup=trailLayer;currentTab="test";drawTrails({trails:[{inventory_index:0,points:[[2,2],[3,3],[4,4]]}]})');
  assert.equal(h.polylines.length,1);assert.deepEqual(Array.from(h.polylines[0].points,point=>Array.from(point)),[[2,2],[3,3],[4,4]]);
  h.nodes.get('test-trail-length').value='0';
  h.run('currentTab="training";drawTrails({trails:[{inventory_index:1,points:[[5,5],[6,6]]}]})');
  assert.equal(h.polylines.length,2,'training paths are independent of the Test path-length field');
  assert.deepEqual(Array.from(h.polylines[1].points,point=>Array.from(point)),[[5,5],[6,6]]);
  h.run('currentTab="mission";drawTrails({trails:[]})');assert.equal(clears,3);
});

test('History drag keeps its range stable and the visible right edge returns to Live',()=>{
  const h=harness();
  h.run('currentTab="training";chart=()=>{};trainingJob={status:"running",timeline:{id:"one",first_step:0,last_step:10,last_seconds:50}};updateTimeline();beginTimelineDrag()');
  assert.equal(Number(h.nodes.get('timeline-slider').max),10);
  h.run('scrubHistory(9);trainingJob.timeline.last_step=12;trainingJob.timeline.last_seconds=60;updateTimeline()');
  assert.equal(Number(h.nodes.get('timeline-slider').max),10,'new steps must not move the drag range');
  assert.equal(h.run('timelineViews.training.step'),9);
  h.run('scrubHistory(10)');
  assert.equal(h.run('timelineViews.training.live'),true,'the frozen right edge follows the live stream');
  h.run('endTimelineDrag()');
  assert.equal(Number(h.nodes.get('timeline-slider').max),12);
  assert.equal(Number(h.nodes.get('timeline-slider').value),12);
  assert.match(h.nodes.get('timeline-time').textContent,/Live · 60 s/);
  h.run('beginTimelineDrag();scrubHistory(5);trainingJob.timeline.last_step=15;endTimelineDrag()');
  assert.equal(Number(h.nodes.get('timeline-slider').value),5,'releasing in history keeps the selected timestamp');
  assert.equal(Number(h.nodes.get('timeline-slider').max),15);
  assert.equal(h.run('timelineViews.training.live'),false);
});

test('Auto-save coalesces edits and serializes in-flight name writes',async()=>{
  const h=harness(),saved=[];let release,active=0,maxActive=0;
  h.context.persist=async value=>{saved.push(value);active++;maxActive=Math.max(active,maxActive);if(value==='B')await new Promise(resolve=>release=resolve);active--;};
  h.run('globalThis.saver=createAutosave("A",persist,error=>{throw error;});saver.change("B")');
  const first=h.run('saver.flush()');
  h.run('saver.change("C");saver.change("D")');await h.run('saver.flush()');
  assert.deepEqual(saved,['B']);release();await first;await Promise.resolve();
  assert.deepEqual(saved,['B','D']);assert.equal(maxActive,1);
});

test('Loaded model is named and unload clears it without sending a training command',async()=>{
  const h=harness(),requests=[];
  h.context.fetch=async(url,options)=>{requests.push({url,options});return {ok:true,json:async()=>({unloaded:true})};};
  h.run('testBackendReady=true;selectedRun={run_id:"one",name:"Square",checkpoint:"best.pt"};updateLoadedName()');
  assert.equal(h.modelOption.textContent,'Square · best.pt');assert.equal(h.modelOption.hidden,false);
  assert.equal(h.run('runIsLoaded("one","best.pt")'),true);
  await h.run('unloadRun()');
  assert.equal(h.run('runIsLoaded("one","best.pt")'),false);assert.equal(h.modelOption.hidden,true);
  assert.deepEqual(requests.map(request=>request.url),['/api/simulation/unload']);
});

test('Rewound training display stays fixed as live data advances; Live restores the newest state',()=>{
  const h=harness(),draws=[];h.context.draws=draws;
  h.run('currentTab="training";applyFrame=frame=>draws.push(frame);applyTelemetry=frame=>draws.push(frame);chart=()=>{};trainingJob={status:"running",phase:"collecting",steps:8,live:{episode:1,episode_step:8,episode_length:500},frame:{tag:"latest-map"},telemetry:{tag:"latest-readings"},timeline:{id:"one",first_step:0,last_step:8,last_seconds:40}};Object.assign(timelineViews.training,{id:"one",live:false,step:2,frame:{episode:1,info:{step:2},history_seconds:10}});renderTraining()');
  assert.equal(draws.length,0);assert.match(h.nodes.get('timeline-time').textContent,/Past · 10 s/);
  h.run('returnLive()');assert.equal(draws.length,2);assert.equal(draws[1].tag,'latest-readings');
  assert.match(h.nodes.get('timeline-time').textContent,/Live · 40 s/);
});

test('Scrubbing requests historical data without resetting or stopping the simulator',async()=>{
  const h=harness(),requests=[];
  h.context.fetch=async url=>{requests.push(url);return {ok:true,json:async()=>({history_step:2,history_seconds:10,episode:1,info:{step:2}})};};
  h.run('currentTab="mission";trainingJob={timeline:{id:"one",first_step:0,last_step:10}};Object.assign(timelineViews.training,{id:"one",live:false,step:2})');
  await h.run('showHistory("training",2)');
  assert.equal(requests.length,1);assert.match(requests[0],/^\/api\/visualization\/history\/training\?/);
  assert.equal(h.run('timelineViews.training.frame.info.step'),2);
});

test('Evaluation has visible progress and persistent results rather than just a toast',()=>{
  const h=harness();
  h.run('currentTab="test";testEvaluation={status:"running",episode:2,episodes:5,step:87,episode_length:500,elapsed_seconds:12};renderEvaluation()');
  assert.equal(h.nodes.get('evaluation-panel').hidden,false);
  assert.match(h.nodes.get('evaluation-progress').textContent,/episode 2 \/ 5 · step 87 \/ 500/);
  assert.equal(h.nodes.get('stop-evaluation').hidden,false);
  h.run('testEvaluation={status:"completed",job_id:"one",elapsed_seconds:15,result:{episodes:5,return_mean:1,coverage_mean:0.5,p10_final:0.1,fairness_final:0.8,energy_Wh:2,crashed_uavs:0}};renderEvaluation()');
  assert.equal(h.nodes.get('evaluation-fill').style.width,'100%');
  assert.match(h.nodes.get('evaluation-result').innerHTML,/Mean return/);
  assert.doesNotMatch(h.nodes.get('evaluation-result').innerHTML,/percentile/i);
  assert.equal(h.nodes.has('metric-p10'),false);
  assert.equal(h.nodes.get('stop-evaluation').hidden,true);
  h.run('updateWorkspaceVisibility()');assert.equal(h.nodes.get('evaluation-panel').hidden,false);
});

for(const policy of ['heuristic','random','model'])test(`Evaluate policy uses the selected ${policy} policy`,async()=>{
  const h=harness(),requests=[];
  h.nodes.get('evaluation-panel').scrollIntoView=()=>{};
  h.nodes.get('simulation-policy').value=policy;
  h.context.fetch=async(url,options)=>{requests.push({url,body:JSON.parse(options.body)});return {ok:true,json:async()=>({status:'running'})};};
  h.run('testBackendReady=true;testSettings=()=>({episodes:3,seed:42,trail_length_m:500});checkFields=async()=>true;renderEvaluation=()=>{};config={scenario:{name:"Mission"}};selectedRun={run_id:"saved",checkpoint:"final.pt"}');
  await h.run('startEvaluation()');
  assert.equal(requests.length,1);
  assert.equal(requests[0].body.policy,policy);
  assert.equal(requests[0].url,policy==='model'?'/api/runs/saved/evaluation/start':'/api/evaluation/start');
  assert.equal(requests[0].body.episodes,3);
  if(policy==='model')assert.equal(requests[0].body.checkpoint,'final.pt');
  else assert.equal(requests[0].body.scenario.name,'Mission');
  const html=fs.readFileSync(path.join(root,'index.html'),'utf8');
  assert.match(html.slice(html.indexOf('id="simulation-controls"')).split('</div>')[0],/id="evaluate-policy"/);
});

test('Training controls distinguish pending pause, paused and one-step processing',async()=>{
  const h=harness(),requests=[];
  for(const [status,label,pauseDisabled,stepDisabled] of [
    ['idle','Pause',true,true],['starting','Pause',true,true],['running','Pause',false,true],
    ['pausing','Pausing…',true,true],['paused','Resume',false,false],
    ['stepping','Pause',true,true],['stopping','Pause',true,true],['completed','Pause',true,true],
  ]){
    h.run(`renderTrainingControls({status:'${status}'})`);
    assert.equal(h.nodes.get('pause-training').textContent,label);
    assert.equal(h.nodes.get('pause-training').disabled,pauseDisabled);
    assert.equal(h.nodes.get('step-training').disabled,stepDisabled);
    assert.equal(h.nodes.get('start-training').disabled,!['idle','completed'].includes(status));
  }
  h.context.fetch=async url=>{requests.push(url);return {ok:true,json:async()=>({status:'stepping'})};};
  h.run('currentTab="mission";trainingJob={status:"paused",steps:7}');
  await h.run('controlTraining("step")');
  assert.deepEqual(requests,['/api/training/step']);
  assert.equal(h.nodes.get('step-training').disabled,true,'prevent a second click while stepping');
  assert.equal(h.run('trainingJob.steps'),7,'only backend telemetry may advance the step count');
  h.run('renderTrainingControls({status:"paused",steps:10,total_steps:10})');
  assert.equal(h.nodes.get('step-training').disabled,true,'final evaluation can resume, with no training steps remaining');
  assert.equal(h.nodes.get('pause-training').disabled,false);
  const html=fs.readFileSync(path.join(root,'index.html'),'utf8');
  assert.match(html,/id="start-training"[\s\S]*?<\/div>\s*<div class="toolbar"><button id="pause-training"[\s\S]*?id="step-training"/);
});

test('Test charts retain per-step samples across episodes and reset for a new playback',()=>{
  const h=harness(),draws=[];h.context.draws=draws;
  h.run('chart=(id,rows,key)=>draws.push({id,steps:rows.map(row=>row.steps),key});trainingHistory=[{steps:999}];currentTab="mission"');
  h.run('recordTestFrame({timeline:{id:"one",last_step:1},episode:1,info:{coverage:{mean:0.2},reward:0.1}});recordTestFrame({timeline:{id:"one",last_step:2},episode:2,info:{coverage:{mean:0.3},reward:0.2}});currentTab="test";renderTestPlots()');
  assert.deepEqual(Array.from(draws[0].steps),[1,2]);
  assert.deepEqual(Array.from(h.run('testHistory.map(row=>row.episode)')),[1,2]);
  assert.equal(draws[0].key,'mean_coverage');assert.equal(draws[1].key,'mean_step_reward');
  h.run('recordTestFrame({timeline:{id:"new",last_step:0},episode:1,info:{coverage:{mean:0},reward:0}})');
  assert.deepEqual(Array.from(h.run('testHistory.map(row=>row.steps)')),[0]);
});

test('Evaluation charts deduplicate polled steps and stay separate from playback and training',()=>{
  const h=harness();
  h.run('currentTab="mission";testPlotSource="evaluation";trainingHistory=[{steps:999}];testHistory=[{steps:888}];acceptEvaluation({job_id:"first",status:"running",samples:[{steps:1,episode:1},{steps:2,episode:1}]});acceptEvaluation({job_id:"first",status:"running",samples:[{steps:2,episode:1},{steps:3,episode:2}]})');
  assert.deepEqual(Array.from(h.run('evaluationHistory.map(row=>row.steps)')),[1,2,3]);
  assert.equal(h.run('trainingHistory[0].steps'),999);assert.equal(h.run('testHistory[0].steps'),888);
  h.run('acceptEvaluation({job_id:"second",status:"running",samples:[{steps:1,episode:1}]})');
  assert.deepEqual(Array.from(h.run('evaluationHistory.map(row=>row.steps)')),[1]);
});

test('Evaluation Pause becomes Continue and preserves graphs when resumed',async()=>{
  const h=harness(),requests=[];
  h.context.fetch=async url=>{requests.push(url);return {ok:true,json:async()=>({status:url.endsWith('/resume')?'running':'paused'})};};
  h.run('currentTab="test";testPlotSource="evaluation";evaluationHistory=[{steps:1,episode:1,mean_coverage:0.2,mean_step_reward:0.1}];testEvaluation={job_id:"one",status:"paused",episode:1,step:1,episodes:2,episode_length:6};renderEvaluation()');
  assert.equal(h.nodes.get('pause-evaluation').textContent,'Continue');
  assert.equal(h.nodes.get('play-simulation').disabled,true);
  await h.run('controlEvaluation("resume")');
  assert.equal(h.nodes.get('pause-evaluation').textContent,'Pause');
  await h.run('controlEvaluation("pause")');
  assert.equal(h.nodes.get('pause-evaluation').textContent,'Continue');
  assert.equal(h.run('evaluationHistory.length'),1);
  assert.deepEqual(requests,['/api/evaluation/resume','/api/evaluation/pause']);
  h.run('testEvaluation={job_id:"one",status:"completed",result:{episodes:0}};renderEvaluation()');
  assert.equal(h.nodes.get('pause-evaluation').hidden,true);
  assert.equal(h.nodes.get('play-simulation').disabled,false);
});
