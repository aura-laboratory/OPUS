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
  return {nodes,context,fits,polylines,run:code=>vm.runInContext(code,context),flush(){while(frames.length)frames.shift()();}};
}

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
  assert.equal(h.nodes.get('charts').hidden,true);
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

test('Test trails use the latest X segments and disappear in other tabs',()=>{
  const h=harness();let clears=0;h.context.trailLayer={clearLayers(){clears++;}};
  h.nodes.get('test-trail-steps').value='2';
  h.run('trailGroup=trailLayer;currentTab="test";drawTrails({trails:[{inventory_index:0,points:[[1,1],[2,2],[3,3],[4,4]]}]})');
  assert.equal(h.polylines.length,1);assert.deepEqual(Array.from(h.polylines[0].points,point=>Array.from(point)),[[2,2],[3,3],[4,4]]);
  h.run('currentTab="mission";drawTrails({trails:[]})');assert.equal(clears,2);
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
  assert.equal(h.nodes.get('stop-evaluation').hidden,true);
  h.run('updateWorkspaceVisibility()');assert.equal(h.nodes.get('evaluation-panel').hidden,false);
});
