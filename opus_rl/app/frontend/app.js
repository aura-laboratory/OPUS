'use strict';
let config, defaults, schema, presets, hardware, map, polygonLayer, previewLayer, heatLayer, markerGroup;
let draft = [], dockPoints = [], mode = 'idle', playing = false, simulationReady = false, currentTab = 'mission';
let activeJob = false, lastFrame, playGeneration = 0, validationTimer, validationVersion = 0;
let trainingJob = null, trainingHistory = [], trainingRun = '', trainingMapKey = '', replayConfig = null, idleTrainingFrame = null;
const viewFrames = {mission:null,runs:null};
const $ = id => document.getElementById(id);
const escapeHTML = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fields = {
  required_gsd_m:['Required GSD','cm/pixel',100], delta_t:['Step time interval ΔT','s'],
  min_altitude_m:['Minimum altitude','m'], max_altitude_m:['Maximum altitude','m'],
  coverage_lambda:['Coverage memory λ',''], horizon:['Episode length','steps'],
  grid_size:['Coverage grid','cells/side'], observation_size:['Observation grid','cells/side'],
  fixed_count:['Number of docks',''],
  max_horizontal_speed_mps:['Horizontal speed','m/s'], max_climb_speed_mps:['Climb speed','m/s'],
  max_descent_speed_mps:['Descent speed','m/s'], max_horizontal_accel_mps2:['Horizontal acceleration','m/s²'],
  max_vertical_accel_mps2:['Vertical acceleration','m/s²'], max_yaw_rate_degps:['Yaw rotational speed','°/s'],
  max_yaw_accel_degps2:['Yaw rotational acceleration','°/s²'],
  max_gimbal_pitch_rate_degps:['Gimbal rotational speed','°/s'],
  gimbal_pitch_min_deg:['Minimum pitch','°'], gimbal_pitch_max_deg:['Maximum pitch','°'],
  battery_capacity_Wh:['Battery capacity','Wh'], reserve_battery_ratio:['Return reserve','%',100],
  P_airborne_base_W:['Airborne power P₀','W'], k_vxy:['Horizontal coefficient kᵥ','W·s²/m²'],
  k_vz_up:['Climb coefficient k↑','W·s/m'], k_vz_down:['Descent coefficient k↓','W·s/m'],
  k_acc:['Acceleration coefficient kₐ','W·s⁴/m²'], takeoff_energy_Wh:['Takeoff energy','Wh'],
  landing_energy_Wh:['Landing energy','Wh'], collision_radius_m:['Effective radius','m'],
  camera_width_px:['Image width','px'], camera_height_px:['Image height','px'],
  camera_fov_deg:['Field of view','°'], charge_power_W:['Charging power','W'],
  total_steps:['Total team steps','steps'],
  rollout_steps:['Update interval','steps'], evaluation_frequency:['Evaluation interval','steps'],
  evaluation_episodes:['Evaluation episodes',''],
  checkpoint_frequency:['Checkpoint interval','steps'], seed:['Seed',''], torch_threads:['CPU threads',''],
  learning_rate:['Learning rate',''], gamma:['Discount γ',''], gae_lambda:['GAE λ',''],
  clip_range:['PPO clip',''], entropy_coef:['Entropy coefficient',''], value_coef:['Value coefficient',''],
  max_grad_norm:['Gradient limit',''], batch_size:['Minibatch size','team steps'], epochs:['Update epochs',''],
  hidden_size:['Network width','neurons'], target_kl:['Target KL',''],
  min_uavs:['Minimum UAVs',''], max_uavs:['Maximum UAVs',''],
  min_stations:['Minimum docks',''], max_stations:['Maximum docks',''],
  mean_weight:['Mean quality',''], fairness_weight:['Fairness',''], p10_weight:['10th percentile',''],
  energy_weight:['Energy',''], projection_weight:['Action projection',''],
  collision_weight:['Collision',''], crash_weight:['Crash','']
};
function notice(message,error=false) {
  $('notice').textContent=message; $('notice').classList.toggle('error',error); $('notice').hidden=false;
  clearTimeout(notice.timer); notice.timer=setTimeout(()=>$('notice').hidden=true,error?12000:4500);
}
function errorPaths(error) {
  let path=(error.loc||[]).filter(x=>x!=='body').join('.'), message=error.msg||String(error);
  const matches = [
    ['Grid size must',['scenario.grid_size','scenario.observation_size']],
    ['Minimum altitude',['scenario.min_altitude_m','scenario.max_altitude_m']],
    ['Coverage weights',['scenario.reward.mean_weight','scenario.reward.fairness_weight','scenario.reward.p10_weight']],
    ['Variable count bounds',['scenario.min_uavs','scenario.max_uavs','scenario.min_stations','scenario.max_stations']],
    ['Place exactly',['scenario.station_positions_wgs84','scenario.station_types.0.fixed_count']],
    ['launch dock',['scenario.uav_launch_docks']],
    ['at least one active',['scenario.station_types.0.fixed_count']]
  ];
  if(message.includes('Fixed UAV count')||message.includes('Fixed station count')) return [path+'.fixed_count'];
  if(message.includes('Gimbal minimum')) return [path+'.gimbal_pitch_min_deg',path+'.gimbal_pitch_max_deg'];
  for(const [text,paths] of matches) if(message.includes(text)) return paths;
  if(!path) return [/station|dock/i.test(message)?'scenario.station_positions_wgs84':'scenario.polygon_wgs84'];
  return [path];
}
function clearErrors() {
  document.querySelectorAll('.field-error').forEach(e=>{e.hidden=true;e.textContent='';});
  document.querySelectorAll('[aria-invalid]').forEach(e=>e.removeAttribute('aria-invalid'));
  document.querySelectorAll('.invalid').forEach(e=>e.classList.remove('invalid'));
}
function showErrors(errors) {
  clearErrors(); let shown=false;
  for(const error of errors) for(const path of errorPaths(error)) {
    const targets=[...document.querySelectorAll('[data-error-path]')].filter(e=>e.dataset.errorPath===path||e.dataset.errorPath.startsWith(path+'.'));
    for(const target of targets) {
      const scale=fields[path.split('.').at(-1)]?.[2]||1;
      target.textContent=(error.msg||String(error)).replace(/^Value error, /,'').replace(/(than(?: or equal to)?) (-?[0-9.]+)/g,(_,text,value)=>text+' '+(Number(value)*scale));
      target.hidden=false; const field=target.closest('.field'); field?.classList.add('invalid');
      field?.querySelector('input,select')?.setAttribute('aria-invalid','true'); shown=true;
    }
  }
  return shown;
}
async function api(path,body) {
  const response=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await response.json();
  if(!response.ok) {
    const errors=Array.isArray(data.detail)?data.detail:[{loc:[],msg:data.detail||'Request failed'}];
    showErrors(errors); throw Error(errors.map(e=>e.msg).join('; '));
  }
  return data;
}
async function checkFields() {
  const version=++validationVersion;
  const result=await api('/api/configuration/check',config);
  if(version===validationVersion) showErrors(result.errors);
  return result.valid;
}
function changed() {
  updateRewardEquation();
  simulationReady=false; clearTimeout(validationTimer);
  validationTimer=setTimeout(()=>checkFields().catch(()=>{}),300);
}
function action(id,fn) {
  $(id).addEventListener('click',async()=>{const button=$(id);button.disabled=true;
    try {await fn();} catch(error) {notice(error.message,true);} finally {button.disabled=false;}
  });
}
function numericField(object,key,path,properties,onChange) {
  const [title,baseUnit,scale=1]=fields[key]||[key,''];
  const unit=key==='torch_threads'&&hardware?'/'+hardware.cpu_threads:baseUnit;
  const definition=properties?.[key]||{}, integer=definition.type==='integer';
  const label=document.createElement('label'); label.className='field'; label.htmlFor=path;
  const text=document.createElement('span'); text.textContent=title; label.append(text);
  const box=document.createElement('span'); box.className='value-box';
  const input=document.createElement('input'); input.id=path; input.type='number';
  input.value=object[key]===null?'':object[key]*scale; input.step=definition.multipleOf|| (integer?'1':'any');
  if(definition.minimum!==undefined) input.min=definition.minimum*scale;
  if(definition.maximum!==undefined) input.max=definition.maximum*scale;
  if(key==='torch_threads'&&hardware)input.max=Math.min(hardware.cpu_threads,definition.maximum||Infinity);
  input.oninput=()=>{const previous=object[key];object[key]=input.value===''?null:Number(input.value)/scale;onChange?.(previous);changed();};
  box.append(input);
  if(unit) {const suffix=document.createElement('span');suffix.className='unit';suffix.textContent=unit;box.append(suffix);}
  label.append(box);
  const error=document.createElement('small');error.className='field-error';error.dataset.errorPath=path;error.id=path+'-error';error.hidden=true;
  input.setAttribute('aria-describedby',error.id);label.append(error);return attachFieldHelp(label,key);
}
function numericFields(container,object,keys,path,properties,onChange) {
  const holder=typeof container==='string'?$(container):container;holder.replaceChildren();
  for(const key of keys) holder.append(numericField(object,key,path+'.'+key,properties,onChange));
}
function details(title,object,keys,path,properties) {
  const group=document.createElement('details'), summary=document.createElement('summary'), holder=document.createElement('div');
  summary.textContent=title;holder.className='fields';numericFields(holder,object,keys,path,properties);
  group.append(summary,holder);return group;
}
function inferPreset(item,options) {
  return options.find(p=>Object.entries(p.values).every(([key,value])=>
    ['name','preset_id','available_count','fixed_count'].includes(key)||item[key]===value))?.id||null;
}
function normalizeMission(s,frame=null) {
  const entries=[],assignments=[];let offset=0;
  const docks=s.station_types[0];
  const count=s.station_mode==='variable'&&frame?frame.stations.length:docks.fixed_count;
  for(const type of s.uav_types) {
    const amount=s.fleet_mode==='variable'?type.available_count:type.fixed_count;
    for(let j=0;j<amount;j++) {
      const entry=structuredClone(type);
      entry.available_count=entry.fixed_count=1;
      entry.preset_id=type.preset_id||inferPreset(type,presets.uavs);
      if(amount>1||presets.uavs.some(p=>p.label===entry.name))entry.name='UAV '+(entries.length+1);
      assignments.push(s.uav_launch_docks?.[offset+j]??(offset+j)%Math.max(1,count));entries.push(entry);
    }
    offset+=type.available_count;
  }
  s.uav_types=entries;s.uav_launch_docks=assignments;s.fleet_mode=s.station_mode='fixed';
  docks.available_count=docks.fixed_count=count;
  docks.preset_id=docks.preset_id||inferPreset(docks,presets.docks);
  if(frame) {
    s.station_positions_wgs84=frame.stations.map(d=>d.geographic);
    s.station_positions_xy=[];s.station_position_types=frame.stations.map(()=>0);
  }
}
function presetField(item,options,title,onSelect) {
  const label=document.createElement('label');label.className='field';label.textContent='Preset';
  const select=document.createElement('select');select.setAttribute('aria-label',title);
  for(const preset of options)select.add(new Option(preset.label,preset.id));
  select.add(new Option('Custom',''));select.value=item.preset_id||'';
  select.onchange=()=>{
    const selected=options.find(p=>p.id===select.value);
    if(selected)onSelect(selected);else{item.preset_id=null;changed();}
  };
  label.append(select);return attachFieldHelp(label,'preset');
}
function nameField(item,index) {
  const path='scenario.uav_types.'+index+'.name',label=document.createElement('label');label.className='field';label.textContent='Name';
  const input=document.createElement('input');input.value=item.name;input.id=path;input.setAttribute('aria-label','Name for UAV '+(index+1));
  input.oninput=()=>{item.name=input.value;changed();};
  const error=document.createElement('small');error.className='field-error';error.dataset.errorPath=path;error.hidden=true;
  label.append(input,error);return attachFieldHelp(label,'name');
}
function renderUAVs() {
  const holder=$('uav-types');holder.replaceChildren();
  const properties=schema.$defs.UAVType.properties;
  config.scenario.uav_types.forEach((item,index)=>{
    const path='scenario.uav_types.'+index,card=document.createElement('article');card.className='uav-card';card.setAttribute('aria-label','UAV '+(index+1));
    const header=document.createElement('div');header.className='card-header';
    const title=document.createElement('h3');title.textContent='UAV '+(index+1);
    const remove=document.createElement('button');remove.className='text-button';remove.textContent='Remove';
    remove.setAttribute('aria-label','Remove UAV '+(index+1));remove.disabled=config.scenario.uav_types.length===1;
    remove.onclick=()=>{
      config.scenario.uav_types.splice(index,1);config.scenario.uav_launch_docks.splice(index,1);
      renderUAVs();changed();
    };
    header.append(title,remove);
    const identity=document.createElement('div');identity.className='identity-fields';
    const preset=presetField(item,presets.uavs,'Preset for UAV '+(index+1),selected=>{
      const name=item.name;Object.assign(item,structuredClone(selected.values),{name});
      renderUAVs();changed();
    });
    const launch=document.createElement('div');launch.className='uav-launch';launch.dataset.uav=index;identity.append(nameField(item,index),launch,preset);
    const propertiesGroup=document.createElement('details'),summary=document.createElement('summary');summary.textContent='Properties';propertiesGroup.append(summary);
    const flight=document.createElement('div');flight.className='fields';
    numericFields(flight,item,['max_horizontal_speed_mps','max_climb_speed_mps','max_descent_speed_mps','max_horizontal_accel_mps2','max_vertical_accel_mps2','max_yaw_rate_degps','max_yaw_accel_degps2','collision_radius_m','battery_capacity_Wh','reserve_battery_ratio'],path,properties);
    propertiesGroup.append(flight,details('Energy model',item,['P_airborne_base_W','k_vxy','k_vz_up','k_vz_down','k_acc','takeoff_energy_Wh','landing_energy_Wh'],path,properties));
    const camera=details('Camera',item,['camera_width_px','camera_height_px','camera_fov_deg','gimbal_pitch_min_deg','gimbal_pitch_max_deg','max_gimbal_pitch_rate_degps'],path,properties);
    const axis=document.createElement('label');axis.className='field';axis.textContent='FOV axis';
    const select=document.createElement('select');select.setAttribute('aria-label','FOV axis for UAV '+(index+1));
    for(const value of ['diagonal','horizontal'])select.add(new Option(value,value));
    select.value=item.camera_fov_axis;select.onchange=()=>{item.camera_fov_axis=select.value;changed();};
    axis.append(select);attachFieldHelp(axis,'camera_fov_axis');camera.append(axis);propertiesGroup.append(camera);
    card.append(header,identity,propertiesGroup);holder.append(card);
  });
  $('add-uav').disabled=config.scenario.uav_types.length>=128;renderLaunchFields();
}
function dockCount() {
  return Math.max(0,Math.min(128,config.scenario.station_types[0].fixed_count||0));
}
function renderLaunchFields() {
  const count=dockCount();
  document.querySelectorAll('.uav-launch').forEach(holder=>{
    holder.replaceChildren();const index=Number(holder.dataset.uav);
    const label=document.createElement('label');label.className='field';label.textContent='Launch dock';
    const select=document.createElement('select');select.setAttribute('aria-label','Launch dock for UAV '+(index+1));
    for(let dock=0;dock<count;dock++)select.add(new Option('Dock '+(dock+1),String(dock)));
    select.value=String(config.scenario.uav_launch_docks?.[index]??index%Math.max(1,count));
    select.onchange=()=>{config.scenario.uav_launch_docks[index]=Number(select.value);changed();};
    const error=document.createElement('small');error.className='field-error';error.dataset.errorPath='scenario.uav_launch_docks.'+index;error.hidden=true;
    label.append(select,error);holder.append(attachFieldHelp(label,'launch_dock'));
  });
}
function renderStation() {
  const holder=$('station-types');holder.replaceChildren();
  const item=config.scenario.station_types[0],properties=schema.$defs.StationType.properties;
  holder.append(presetField(item,presets.docks,'Charging station preset',selected=>{
    const count=item.fixed_count;Object.assign(item,structuredClone(selected.values),{available_count:count,fixed_count:count});
    renderStation();changed();
  }));
  const row=document.createElement('div');row.className='fields';
  row.append(numericField(item,'fixed_count','scenario.station_types.0.fixed_count',properties,()=>{
    if(Number.isInteger(item.fixed_count)&&item.fixed_count>=0&&item.fixed_count<=128) {
      item.available_count=Math.max(1,item.fixed_count);
      if(dockPoints.length>item.fixed_count){dockPoints.length=item.fixed_count;syncDocks();}
      renderLaunchFields();
    }
  }),numericField(item,'charge_power_W','scenario.station_types.0.charge_power_W',properties));
  holder.append(row);
}
function renderForms() {
  const s=config.scenario,properties=schema.$defs.ScenarioConfig.properties;
  numericFields('mission-fields',s,['required_gsd_m','min_altitude_m','max_altitude_m','coverage_lambda','grid_size','observation_size'],'scenario',properties);
  renderUAVs();renderStation();
  const main=['total_steps','rollout_steps','evaluation_frequency','evaluation_episodes','checkpoint_frequency','seed','torch_threads'];
  numericFields('training-fields',s,['delta_t','horizon'],'scenario',properties);
  for(const key of main)$('training-fields').append(numericField(config.training,key,'training.'+key,schema.$defs.TrainConfig.properties));
  const device=document.createElement('label');device.className='field';device.textContent='Compute device';
  const select=document.createElement('select');select.id='training.device';
  select.add(new Option('Auto','auto'));select.add(new Option('CPU','cpu'));
  const gpu=new Option(hardware?.cuda_available?'GPU':'GPU (unavailable)','cuda');gpu.disabled=!hardware?.cuda_available;select.add(gpu);
  select.value=config.training.device;select.onchange=()=>{config.training.device=select.value;changed();};device.append(select);
  const deviceError=document.createElement('small');deviceError.className='field-error';deviceError.dataset.errorPath='training.device';deviceError.id='training.device-error';deviceError.hidden=true;device.append(deviceError);
  select.setAttribute('aria-describedby',deviceError.id);
  const detected=hardware?.gpu_names?.length?'Detected GPU: '+hardware.gpu_names.join(', ')+'.':'';
  $('training-fields').append(attachFieldHelp(device,'device',detected));
  numericFields('ppo-fields',config.training,Object.keys(config.training).filter(k=>!main.includes(k)&&!['device','evaluation_horizon','num_envs'].includes(k)),'training',schema.$defs.TrainConfig.properties);
  numericFields('reward-fields',s.reward,Object.keys(s.reward),'scenario.reward',schema.$defs.RewardConfig.properties);
  updateRewardEquation();
}
function updateRewardEquation() {
  document.querySelectorAll('[data-reward-weight]').forEach(node=>{
    const value=config?.scenario.reward[node.dataset.rewardWeight];node.textContent=Number.isFinite(value)?String(value):'?';
  });
}
function setTab(tab) {
  if(tab!==currentTab){pause();simulationReady=false;if(mode==='draw')cancelDrawing();setMode('idle');}
  currentTab=tab;document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('selected',b.dataset.tab===tab));
  for(const key of ['mission','training','runs'])$(key+'-controls').hidden=key!==tab;
  updateWorkspaceVisibility();
  if(tab==='training'){trainingMapKey='';renderTraining();}
  else if(viewFrames[tab])applyFrame(viewFrames[tab]);
  map?.invalidateSize();
  if(tab==='runs')loadRuns().catch(e=>notice(e.message,true));
}
function updateWorkspaceVisibility() {
  document.querySelectorAll('[data-tabs]').forEach(element=>element.hidden=!element.dataset.tabs.split(' ').includes(currentTab));
  const emptyRun=currentTab==='runs'&&!viewFrames.runs;
  $('runs-placeholder').hidden=!emptyRun;
  $('map').hidden=emptyRun||!map;$('map-fallback').hidden=emptyRun||Boolean(map);
  for(const id of ['simulation-bar','live-metrics','safety-metrics','fleet-status'])$(id).hidden=emptyRun;
  $('simulation-label').textContent=currentTab==='training'?'Training episode':currentTab==='runs'?'Saved model replay':'Mission preview';
}
function setMode(value) {
  mode=value;
  $('draw-area').setAttribute('aria-pressed',String(value==='draw'));
  $('place-station').setAttribute('aria-pressed',String(value==='dock'));
  $('map-mode').textContent=value==='draw'?'Draw area':value==='dock'?'Place docks':'';
  if(value!=='draw'){draft=[];previewLayer?.remove();previewLayer=null;}
  if(map)map.getContainer().style.cursor=value==='idle'?'':'crosshair';
}
function cancelDrawing() {
  const points=config.scenario.polygon_wgs84||lastFrame?.polygon_wgs84;
  if(points)drawPolygon(points);
}
function drawPolygon(points) {
  if(!map)return;if(polygonLayer)polygonLayer.remove();
  polygonLayer=L.polygon(points.map(p=>[p[1],p[0]]),{color:'#dc9d49',weight:2,fillColor:'#dc9d49',fillOpacity:.1}).addTo(map);
}
const uavSymbol='<svg viewBox="0 0 24 24" aria-hidden="true"><g fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><circle cx="5.5" cy="5.5" r="3.4"/><circle cx="18.5" cy="5.5" r="3.4"/><circle cx="5.5" cy="18.5" r="3.4"/><circle cx="18.5" cy="18.5" r="3.4"/><path d="m8 8 8 8m0-8-8 8"/><rect x="9" y="8" width="6" height="8" rx="2" fill="currentColor"/></g></svg>';
function uavIcon(index,status,atDock){return L.divIcon({className:'uav-marker '+status,html:uavSymbol+'<span class="uav-number">'+(index+1)+'</span>',iconSize:[32,32],iconAnchor:atDock?[-25,16]:[16,16]});}
const batteryIcon='<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 2h6v2h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2V2Zm-2 4v14h10V6H7Zm2 7 3-5v4h3l-3 6v-5H9Z"/></svg>';
function dockIcon(index) {return L.divIcon({className:'dock-marker',html:batteryIcon+'<span class="dock-number">'+(index+1)+'</span>',iconSize:[30,30],iconAnchor:[15,15]});}
function drawDocks() {if(!markerGroup)return;markerGroup.clearLayers();dockPoints.forEach((point,index)=>L.marker([point.geographic[1],point.geographic[0]],{icon:dockIcon(index)}).addTo(markerGroup).bindTooltip('Dock '+(index+1)));}
function syncDocks() {
  config.scenario.station_positions_wgs84=dockPoints.map(p=>p.geographic);
  config.scenario.station_position_types=dockPoints.map(()=>0);config.scenario.station_positions_xy=[];
  renderLaunchFields();drawDocks();changed();
}
function uavLabel(u){const id='UAV '+((u.inventory_index??u.id)+1);return id+(u.type&&u.type!==id?' · '+u.type:'');}
function applyTelemetry(frame){const c=frame.info.coverage;$('safety-metrics').textContent=`${frame.info.energy_Wh.reduce((a,b)=>a+b,0).toFixed(3)} Wh this step · ${frame.info.crashed_uavs} crashed · ${frame.info.collision_pairs.length} collision pairs`;$('metric-mean').textContent=c.mean.toFixed(3);$('metric-p10').textContent=c.p10.toFixed(3);$('metric-fairness').textContent=c.fairness.toFixed(3);$('metric-reward').textContent=Number(frame.info.reward||0).toFixed(3);$('simulation-time').textContent=frame.info.simulated_seconds.toLocaleString()+' s';
  $('reward-label').textContent=frame.info.reward_mode==='individual'?'Mean UAV reward':'Team reward';
  $('metric-reward').parentElement.dataset.help=frame.info.reward_mode==='individual'?fieldHelp.live_reward:'Team reward for the latest step, using this legacy MAPPO model’s original reward definition. This is not a cumulative episode return.';
  const fleet=$('fleet-status'),key=frame.uavs.map(u=>u.inventory_index??u.id).join(',');
  if(fleet.dataset.fleet!==key){fleet.dataset.fleet=key;fleet.innerHTML=frame.uavs.map(()=>`<div class="uav-status" data-help="${escapeHTML(fieldHelp.fleet_reading)}" tabindex="0"><span class="status-dot"></span><strong></strong><span></span></div>`).join('');}
  frame.uavs.forEach((u,index)=>{const row=fleet.children[index];row.children[0].className='status-dot '+u.status;row.children[1].textContent=uavLabel(u);row.children[2].textContent=`${(u.battery_ratio*100).toFixed(1)}% · ${u.status}${Number.isFinite(u.reward)?' · reward '+u.reward.toFixed(3):''}`;});
}
function applyFrame(frame,fit=false){lastFrame=frame;if(currentTab!=='training')viewFrames[currentTab]=frame;applyTelemetry(frame);if(currentTab==='mission')$('area-summary').textContent=`${(frame.area_m2/10000).toFixed(2)} ha · ${frame.cell_size_m.toFixed(3)} m cells`;
  if(!map)return;drawPolygon(frame.polygon_wgs84);if(fit)map.fitBounds(polygonLayer.getBounds().pad(.2));markerGroup.clearLayers();for(const [i,s] of frame.stations.entries()){L.marker([s.geographic[1],s.geographic[0]],{icon:dockIcon(i)}).addTo(markerGroup).bindTooltip(`Dock ${i+1}${s.reserved!==null?' · reserved':''}${s.occupied!==null?' · charging':''}`);}for(const u of frame.uavs){if(u.status==='queued')continue;L.marker([u.geographic[1],u.geographic[0]],{icon:uavIcon(u.inventory_index??u.id,u.status,frame.stations.some(s=>Math.hypot(s.position[0]-u.position[0],s.position[1]-u.position[1])<1))}).addTo(markerGroup).bindTooltip(`${escapeHTML(uavLabel(u))} · ${u.position[2].toFixed(1)} m · ${Math.round(u.battery_ratio*100)}%`);}
  const size=frame.heatmap.length,canvas=document.createElement('canvas');canvas.width=size;canvas.height=size;const ctx=canvas.getContext('2d'),pixels=ctx.createImageData(size,size);for(let row=0;row<size;row++)for(let col=0;col<size;col++){const value=frame.heatmap[row][col],offset=((size-1-row)*size+col)*4;pixels.data[offset]=235;pixels.data[offset+1]=165;pixels.data[offset+2]=65;pixels.data[offset+3]=Math.round(Math.min(1,value*4)*170*frame.heatmap_mask[row][col]);}ctx.putImageData(pixels,0,0);if(heatLayer)heatLayer.remove();heatLayer=L.imageOverlay(canvas.toDataURL(),frame.bounds_wgs84.map(p=>[p[1],p[0]]),{opacity:.8,interactive:false}).addTo(map);polygonLayer.bringToFront();}
async function validate(){if(!await checkFields())throw Error('Correct the highlighted fields.');const result=await api('/api/validate',config);idleTrainingFrame=result.frame;viewFrames.mission=result.frame;setMode('idle');applyFrame(result.frame,true);return result;}
function pause(){playing=false;playGeneration++;$('play-simulation').textContent='Play';}
async function reset(stopPlay=true){const tab=currentTab;if(tab==='training')return;if(tab==='mission'&&!await checkFields())throw Error('Correct the highlighted fields.');if(stopPlay)pause();setMode('idle');const frame=await api('/api/simulation/reset',tab==='runs'?replayConfig:config);viewFrames[tab]=frame;if(currentTab===tab){simulationReady=true;applyFrame(frame,true);}}
async function step(){const tab=currentTab;if(tab==='training')return;if(!simulationReady)await reset(false);if(currentTab!==tab)return;const frame=await api('/api/simulation/step',{policy:tab==='runs'?'model':$('simulation-policy').value});viewFrames[tab]=frame;if(currentTab===tab)applyFrame(frame);}
async function play(){if(playing){pause();return;}playing=true;const generation=++playGeneration;$('play-simulation').textContent='Pause';while(playing&&generation===playGeneration){try{await step();}catch(e){notice(e.message,true);pause();}if(playing&&generation===playGeneration)await new Promise(r=>setTimeout(r,120));}}
function diagnostics(metrics) {
  if(!metrics){$('training-diagnostics').innerHTML='<p class="muted">Available after the first rollout.</p>';return;}
  const entries=[['Policy loss','actor_loss'],['Critic loss','value_loss'],['Gaussian entropy','entropy'],['Approximate KL','approximate_kl'],['Explained variance','explained_variance'],['Coverage actor fraction','actor_sample_fraction'],['Energy · Wh/decision','energy_Wh_per_step'],['Projection fraction','reward_projection'],['Collision fraction','reward_collision'],['Crashed fraction','reward_crash']];
  const holder=$('training-diagnostics');
  if(!holder.querySelector('dl'))holder.innerHTML='<dl>'+entries.map(([label,key])=>`<div tabindex="0" data-help="${escapeHTML(fieldHelp[key]||'')}"><dt>${label}</dt><dd></dd></div>`).join('')+'</dl>';
  holder.querySelectorAll('dd').forEach((node,index)=>node.textContent=Number(metrics[entries[index][1]]).toFixed(3));
}
const timingLabels={initialization:'Initialization',policy:'Policy inference',simulation:'Simulation',bootstrap:'Next-state value',buffer:'Buffer / bookkeeping',reset:'Episode reset',update:'Weight update',evaluation:'Evaluation',checkpoint:'Checkpoint saving',logging:'Logging',telemetry:'Live reporting','simulation.motion_safety':'Motion / safety','simulation.camera':'Camera','simulation.coverage_reward':'Coverage / rewards','simulation.observation':'Observation'};
function timingDiagnostics(rows) {
  const entries=Object.entries(rows||{});
  if(!entries.length){delete $('timing-diagnostics').dataset.phases;$('timing-diagnostics').innerHTML='<p class="muted">Available during training.</p>';return;}
  const keys=Object.keys(timingLabels).filter(key=>rows[key]&&!key.includes('.'));
  const display=keys.flatMap(key=>key==='simulation'?[key,...Object.keys(timingLabels).filter(name=>name.startsWith('simulation.')&&rows[name])]:[key]);
  const holder=$('timing-diagnostics'),signature=display.join(',');
  if(holder.dataset.phases!==signature){holder.dataset.phases=signature;holder.innerHTML='<table><thead><tr><th>Phase</th><th tabindex="0" data-help="Total wall-clock seconds spent in this phase since training began.">Total s</th><th tabindex="0" data-help="Mean milliseconds per call. Simulation and inference are called per step; weight updates and evaluation run less often.">ms/call</th><th tabindex="0" data-help="Share of measured top-level processing time. Indented simulator rows are parts of Simulation and are not added to the total twice.">%</th></tr></thead><tbody>'+display.map(key=>`<tr tabindex="0" class="${key.includes('.')?'subphase':''}"><td>${timingLabels[key]}</td><td></td><td></td><td></td></tr>`).join('')+'</tbody></table>';}
  holder.querySelectorAll('tbody tr').forEach((node,index)=>{const row=rows[display[index]];node.dataset.help=`${row.calls} calls · last ${(1000*row.last_seconds).toFixed(2)} ms · slowest ${(1000*row.max_seconds).toFixed(2)} ms`;node.children[1].textContent=row.seconds.toFixed(2);node.children[2].textContent=row.mean_ms.toFixed(2);node.children[3].textContent=row.percent.toFixed(1);});
}
function chart(id,history,key,color){
  const canvas=$(id),ratio=window.devicePixelRatio||1,w=canvas.clientWidth;
  if(!w)return;
  canvas.width=w*ratio;canvas.height=170*ratio;const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);ctx.clearRect(0,0,w,170);
  if(!history.length)return;
  const values=history.map(x=>x[key]),min=Math.min(0,...values),max=Math.max(.01,...values),first=history[0].steps,last=history.at(-1).steps;
  ctx.strokeStyle='#344c5e';ctx.fillStyle='#9ab0c0';ctx.font='11px Segoe UI';
  for(let i=0;i<3;i++){const y=15+i*58;ctx.beginPath();ctx.moveTo(45,y);ctx.lineTo(w-8,y);ctx.stroke();ctx.fillText((max-(max-min)*i/2).toFixed(2),3,y+3);}
  ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.beginPath();
  history.forEach((point,index)=>{const x=45+(w-58)*(point.steps-first)/Math.max(1,last-first),y=15+116*(1-(point[key]-min)/(max-min||1));if(index&&point.episode===history[index-1].episode&&point.steps===history[index-1].steps+1)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.stroke();
  ctx.fillStyle=color;
  history.forEach((point,index)=>{if(index&&point.episode===history[index-1].episode)return;const x=45+(w-58)*(point.steps-first)/Math.max(1,last-first),y=15+116*(1-(point[key]-min)/(max-min||1));ctx.beginPath();ctx.arc(x,y,2,0,Math.PI*2);ctx.fill();});
  ctx.fillStyle='#9ab0c0';ctx.fillText(first.toLocaleString(),45,160);const label=last.toLocaleString()+' steps';ctx.fillText(label,Math.max(45,w-ctx.measureText(label).width-8),160);
}
function renderTraining(){
  const job=trainingJob;
  if(!job)return;
  if(job.status==='idle'&&idleTrainingFrame){if(trainingMapKey!=='idle'){applyFrame(idleTrainingFrame);trainingMapKey='idle';}$('episode-counter').textContent='';$('training-phase').textContent='No experiment running.';chart('coverage-chart',[],'mean_coverage','#e9b36c');chart('reward-chart',[],'mean_step_reward','#8cb9cd');return;}
  if(job.frame){const key=(job.run_id||'starting')+':'+job.frame_step;if(trainingMapKey!==key){applyFrame(job.frame);trainingMapKey=key;}}
  if(job.telemetry)applyTelemetry(job.telemetry);
  const live=job.live;
  $('episode-counter').textContent=live?`Episode ${live.episode} · step ${live.episode_step} / ${live.episode_length}`:'Episode 1 · step 0';
  const phases={starting:'Starting',collecting:'Collecting experience',updating:'Updating weights',evaluating:'Evaluating',checkpointing:'Saving checkpoint',completed:'Completed',stopped:'Stopped',failed:'Failed'};
  let phase=activeJob?(phases[job.phase]||'Starting'):(phases[job.status]||'');
  const evaluation=job.evaluation_progress;
  if(job.phase==='evaluating'&&activeJob&&evaluation)phase+=` · episode ${evaluation.episode} / ${evaluation.episodes} · step ${evaluation.step} / ${evaluation.episode_length}`;
  $('training-phase').textContent=phase;
  chart('coverage-chart',trainingHistory,'mean_coverage','#e9b36c');chart('reward-chart',trainingHistory,'mean_step_reward','#8cb9cd');
}
async function poll(){
  try{
    const cursor=trainingHistory.at(-1)?.steps??-1;
    const job=await api(`/api/training/status?after_step=${cursor}&run_id=${encodeURIComponent(trainingRun)}`);
    if((job.run_id||'')!==trainingRun||job.steps<(trainingHistory.at(-1)?.steps||0)){
      trainingHistory=[];trainingRun=job.run_id||'';trainingMapKey='';
    }
    trainingJob=job;
    const previousStep=trainingHistory.at(-1)?.steps??-1;
    trainingHistory.push(...(job.samples||[]).filter(row=>row.steps>previousStep));trainingHistory=trainingHistory.slice(-2000);
    activeJob=['starting','running','stopping'].includes(job.status);
    $('job-badge').textContent=job.status==='idle'?'Ready':job.status[0].toUpperCase()+job.status.slice(1);
    $('connection-status').textContent='Local backend connected';$('start-training').disabled=activeJob;$('stop-training').disabled=!activeJob;
    if(job.total_steps){$('progress-fill').style.width=Math.min(100,100*job.steps/job.total_steps)+'%';$('progress-text').textContent=`${(job.steps||0).toLocaleString()} / ${job.total_steps.toLocaleString()} steps · ${Number(job.steps_per_second||0).toFixed(1)} steps/s`;}
    if(currentTab==='training')renderTraining();
    diagnostics(job.metrics);timingDiagnostics(job.timings);
    if(job.status==='failed'&&job.error!==poll.lastError){notice(job.error,true);poll.lastError=job.error;}
  }catch(e){$('connection-status').textContent='Backend disconnected';}
  setTimeout(poll,500);
}
async function loadRuns(){
  const runs=await api('/api/runs');$('run-list').innerHTML='';
  if(!runs.length){$('run-list').innerHTML='<p class="muted">No runs yet. Start an experiment in Training.</p>';return;}
  for(const run of runs){
    const item=document.createElement('div');item.className='run-block';const evaluation=run.evaluation;
    item.innerHTML=`<h3>${escapeHTML(run.run_id)}</h3><p>${escapeHTML(run.algorithm||'mappo').toUpperCase()} · ${escapeHTML(run.status)} · ${run.steps.toLocaleString()} steps · seed ${run.seed}<br>${evaluation?'Last evaluation return '+evaluation.return_mean.toFixed(2)+' · mean quality '+evaluation.coverage_mean.toFixed(3):'No scheduled evaluation recorded.'}</p><label>Checkpoint<select>${run.checkpoints.map(c=>`<option ${c==='best.pt'?'selected':''}>${escapeHTML(c)}</option>`).join('')}</select></label><div class="toolbar"><button class="replay">Replay</button><button class="evaluate">Evaluate</button><a href="/api/runs/${encodeURIComponent(run.run_id)}/export">Export ZIP</a></div>`;
    item.querySelector('.replay').onclick=async()=>{
      try{
        pause();const checkpoint=item.querySelector('select').value;
        const result=await api(`/api/runs/${encodeURIComponent(run.run_id)}/load`,{checkpoint});
        replayConfig=result.config;simulationReady=currentTab==='runs';viewFrames.runs=result.frame;
        $('loaded-model').textContent=`${run.run_id} · ${checkpoint}`;
        if(currentTab==='runs'){updateWorkspaceVisibility();applyFrame(result.frame,true);}
      }catch(e){notice(e.message,true);}
    };
    item.querySelector('.evaluate').onclick=async(event)=>{
      event.target.disabled=true;pause();
      try{
        const checkpoint=item.querySelector('select').value;
        const result=await api(`/api/runs/${encodeURIComponent(run.run_id)}/evaluate`,{checkpoint,episodes:config.training.evaluation_episodes});
        // Load the same checkpoint for subsequent replay controls.
        const loaded=await api(`/api/runs/${encodeURIComponent(run.run_id)}/load`,{checkpoint});
        replayConfig=loaded.config;simulationReady=false;viewFrames.runs=result.frame;
        $('loaded-model').textContent=`${run.run_id} · ${checkpoint}`;
        if(currentTab==='runs'){updateWorkspaceVisibility();applyFrame(result.frame,true);}
        notice(`Evaluation: return ${result.return_mean.toFixed(2)} · mean quality ${result.coverage_mean.toFixed(3)} · crashes ${result.crashed_uavs.toFixed(1)}`);
      }catch(e){notice(e.message,true);}finally{event.target.disabled=false;}
    };
    attachFieldHelp(item.querySelector('select'),'checkpoint');$('run-list').append(item);
  }
}

async function initialize() {
  try {
    const [response,area]=await Promise.all([api('/api/defaults'),api('/static/default-area.json')]);
    defaults=response.config;schema=response.schema;presets=response.presets;hardware=response.hardware||null;config=structuredClone(defaults);
    Object.assign(config.scenario,area);normalizeMission(config.scenario);renderForms();
    if(window.L) {
      map=L.map('map',{doubleClickZoom:false}).setView([...config.scenario.map_origin_wgs84].reverse(),18);
      L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'}).addTo(map);
      markerGroup=L.layerGroup().addTo(map);
      map.on('click',event=>{
        if(mode==='draw'){draft.push([event.latlng.lng,event.latlng.lat]);drawPolygon(draft);}
        else if(mode==='dock'){if(dockPoints.length>=dockCount()){notice('All docks are placed.');return;}dockPoints.push({type:0,geographic:[event.latlng.lng,event.latlng.lat]});syncDocks();}
      });
      map.on('mousemove',event=>{
        if(mode==='draw'&&draft.length){previewLayer?.remove();previewLayer=L.polyline([[draft.at(-1)[1],draft.at(-1)[0]],event.latlng],{color:'#e4b46d',weight:1,dashArray:'5 6'}).addTo(map);}
      });
      map.on('contextmenu',()=>{
        if(mode==='draw'&&draft.length>=3){
          config.scenario.polygon_wgs84=[...draft];dockPoints=[];heatLayer?.remove();syncDocks();simulationReady=false;setMode('dock');
        }
      });
    } else {$('map').hidden=true;$('map-fallback').hidden=false;}
    const initial=await api('/api/validate',config);idleTrainingFrame=initial.frame;applyFrame(initial.frame,true);
    dockPoints=initial.frame.stations.map(s=>({type:0,geographic:s.geographic}));
    document.querySelectorAll('nav button').forEach(button=>button.onclick=()=>setTab(button.dataset.tab));
    action('validate',validate);action('reset-simulation',()=>reset());action('step-simulation',step);$('play-simulation').onclick=play;
    $('draw-area').onclick=()=>{
      pause();
      if(mode==='draw'){cancelDrawing();setMode('idle');return;}
      draft=[];simulationReady=false;setMode('draw');
    };
    $('place-station').onclick=()=>{
      pause();if(mode==='draw')cancelDrawing();setMode(mode==='dock'?'idle':'dock');
    };
    action('clear-docks',()=>{dockPoints=[];syncDocks();});
    action('add-uav',()=>{
      config.scenario.uav_types.push({...structuredClone(presets.uavs[0].values),name:'UAV '+(config.scenario.uav_types.length+1)});
      config.scenario.uav_launch_docks.push(0);renderUAVs();changed();
    });
    action('start-training',async()=>{
      const health=await api('/api/health');
      if(health.training_algorithm!=='ippo'||health.training_controls_revision!==4)throw Error('Restart OPUS to load live training readings and timing diagnostics.');
      pause();config.scenario.reward_mode='individual';await validate();await api('/api/training/start',config);
    });
    action('stop-training',async()=>{await api('/api/training/stop',{});});
    action('refresh-runs',loadRuns);
    $('search-form').onsubmit=async event=>{
      event.preventDefault();try {
        const found=await api('/api/geocode?q='+encodeURIComponent($('place-search').value));$('search-results').replaceChildren();
        for(const place of found){
          const button=document.createElement('button');button.textContent=place.display_name;
          button.onclick=()=>{map?.setView([Number(place.lat),Number(place.lon)],16);$('search-results').replaceChildren();};$('search-results').append(button);
        }
        if(!found.length)notice('No places found.');
      } catch(error){notice(error.message,true);}
    };
    updateWorkspaceVisibility();poll();
  } catch(error){notice(error.message,true);$('connection-status').textContent='Unable to initialize';}
}
initializeFieldHelp();
attachFieldHelp($('place-search'),'place_search');attachFieldHelp($('simulation-policy'),'simulation_policy');
document.querySelectorAll('[data-reading-help]').forEach(element=>{attachFieldHelp(element,element.dataset.readingHelp);element.tabIndex=0;});
attachFieldHelp($('safety-metrics'),'safety_reading');attachFieldHelp($('training-phase'),'phase_reading');
initialize();
