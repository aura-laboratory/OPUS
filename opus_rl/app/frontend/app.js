'use strict';
let config, defaults, schema, presets, map, polygonLayer, previewLayer, heatLayer, markerGroup;
let draft = [], dockPoints = [], mode = 'idle', playing = false, simulationReady = false, currentTab = 'mission';
let activeJob = false, lastFrame, playGeneration = 0, validationTimer, validationVersion = 0;
const $ = id => document.getElementById(id);
const escapeHTML = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fields = {
  required_gsd_m:['Required GSD','cm/pixel',100], delta_t:['Step time interval ΔT','s'],
  min_altitude_m:['Minimum altitude','m'], max_altitude_m:['Maximum altitude','m'],
  coverage_lambda:['Coverage memory λ',''], horizon:['Episode steps','steps'],
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
  total_steps:['Total team steps','steps'], num_envs:['Parallel environments',''],
  rollout_steps:['Rollout length','steps'], evaluation_frequency:['Evaluation interval','steps'],
  evaluation_episodes:['Evaluation episodes',''], evaluation_horizon:['Evaluation length','steps'],
  checkpoint_frequency:['Checkpoint interval','steps'], seed:['Seed',''], torch_threads:['CPU threads',''],
  learning_rate:['Learning rate',''], gamma:['Discount γ',''], gae_lambda:['GAE λ',''],
  clip_range:['PPO clip',''], entropy_coef:['Entropy coefficient',''], value_coef:['Value coefficient',''],
  max_grad_norm:['Gradient limit',''], batch_size:['Batch size',''], epochs:['Update epochs',''],
  hidden_size:['Network width',''], target_kl:['Target KL',''],
  min_uavs:['Minimum UAVs',''], max_uavs:['Maximum UAVs',''],
  min_stations:['Minimum docks',''], max_stations:['Maximum docks',''],
  mean_weight:['Mean quality',''], fairness_weight:['Fairness',''], p10_weight:['Lowest 10%',''],
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
  simulationReady=false; clearTimeout(validationTimer);
  validationTimer=setTimeout(()=>checkFields().catch(()=>{}),300);
}
function action(id,fn) {
  $(id).addEventListener('click',async()=>{const button=$(id);button.disabled=true;
    try {await fn();} catch(error) {notice(error.message,true);} finally {button.disabled=false;}
  });
}
function numericField(object,key,path,properties,onChange) {
  const [title,unit,scale=1]=fields[key]||[key,''];
  const definition=properties?.[key]||{}, integer=definition.type==='integer';
  const label=document.createElement('label'); label.className='field'; label.htmlFor=path;
  const text=document.createElement('span'); text.textContent=title; label.append(text);
  const box=document.createElement('span'); box.className='value-box';
  const input=document.createElement('input'); input.id=path; input.type='number';
  input.value=object[key]===null?'':object[key]*scale; input.step=integer?'1':'any';
  if(definition.minimum!==undefined) input.min=definition.minimum*scale;
  if(definition.maximum!==undefined) input.max=definition.maximum*scale;
  input.oninput=()=>{const previous=object[key];object[key]=input.value===''?null:Number(input.value)/scale;onChange?.(previous);changed();};
  box.append(input);
  if(unit) {const suffix=document.createElement('span');suffix.className='unit';suffix.textContent=unit;box.append(suffix);}
  label.append(box);
  const error=document.createElement('small');error.className='field-error';error.dataset.errorPath=path;error.id=path+'-error';error.hidden=true;
  input.setAttribute('aria-describedby',error.id);label.append(error);return label;
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
  label.append(select);return label;
}
function nameField(item,index) {
  const path='scenario.uav_types.'+index+'.name',label=document.createElement('label');label.className='field';label.textContent='Name';
  const input=document.createElement('input');input.value=item.name;input.id=path;input.setAttribute('aria-label','Name for UAV '+(index+1));
  input.oninput=()=>{item.name=input.value;changed();};
  const error=document.createElement('small');error.className='field-error';error.dataset.errorPath=path;error.hidden=true;
  label.append(input,error);return label;
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
    axis.append(select);camera.append(axis);propertiesGroup.append(camera);
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
    label.append(select,error);holder.append(label);
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
  numericFields('mission-fields',s,['required_gsd_m','min_altitude_m','max_altitude_m','coverage_lambda'],'scenario',properties);
  numericFields('episode-fields',s,['delta_t','horizon'],'scenario',properties);
  numericFields('grid-fields',s,['grid_size','observation_size'],'scenario',properties);
  renderUAVs();renderStation();
  const main=['total_steps','num_envs','rollout_steps','evaluation_frequency','evaluation_episodes','evaluation_horizon','checkpoint_frequency','seed'];
  numericFields('training-fields',config.training,main,'training',schema.$defs.TrainConfig.properties);
  numericFields('ppo-fields',config.training,Object.keys(config.training).filter(k=>!main.includes(k)&&k!=='device'),'training',schema.$defs.TrainConfig.properties);
  numericFields('reward-fields',s.reward,Object.keys(s.reward),'scenario.reward',schema.$defs.RewardConfig.properties);
}
function setTab(tab) {
  currentTab=tab;document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('selected',b.dataset.tab===tab));
  for(const key of ['mission','training','runs'])$(key+'-controls').hidden=key!==tab;
  $('charts').hidden=tab!=='training';map?.invalidateSize();
  if(tab==='runs')loadRuns().catch(e=>notice(e.message,true));
}
function setMode(value) {mode=value;$('map-mode').textContent=value==='draw'?'Draw area':value==='dock'?'Place docks':'';if(map)map.getContainer().style.cursor=value==='idle'?'':'crosshair';}
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
function applyFrame(frame,fit=false){lastFrame=frame;const c=frame.info.coverage;$('safety-metrics').textContent=`${frame.info.energy_Wh.reduce((a,b)=>a+b,0).toFixed(3)} Wh this decision · ${frame.info.crashed_uavs} crashed · ${frame.info.collision_pairs.length} collision pairs`;$('metric-mean').textContent=c.mean.toFixed(3);$('metric-p10').textContent=c.p10.toFixed(3);$('metric-fairness').textContent=c.fairness.toFixed(3);$('metric-reward').textContent=Number(frame.info.reward||0).toFixed(3);$('simulation-time').textContent=frame.info.simulated_seconds.toLocaleString()+' s';$('area-summary').textContent=`${(frame.area_m2/10000).toFixed(2)} ha · ${frame.cell_size_m.toFixed(3)} m cells`;
  $('fleet-status').innerHTML=frame.uavs.map(u=>`<div class="uav-status"><span class="status-dot ${escapeHTML(u.status)}"></span><strong>${escapeHTML(uavLabel(u))}</strong><span>${Math.round(u.battery_ratio*100)}% · ${escapeHTML(u.status)}</span></div>`).join('');
  if(!map)return;drawPolygon(frame.polygon_wgs84);if(fit)map.fitBounds(polygonLayer.getBounds().pad(.2));markerGroup.clearLayers();for(const [i,s] of frame.stations.entries()){L.marker([s.geographic[1],s.geographic[0]],{icon:dockIcon(i)}).addTo(markerGroup).bindTooltip(`Dock ${i+1}${s.reserved!==null?' · reserved':''}${s.occupied!==null?' · charging':''}`);}for(const u of frame.uavs){if(u.status==='queued')continue;L.marker([u.geographic[1],u.geographic[0]],{icon:uavIcon(u.inventory_index??u.id,u.status,frame.stations.some(s=>Math.hypot(s.position[0]-u.position[0],s.position[1]-u.position[1])<1))}).addTo(markerGroup).bindTooltip(`${escapeHTML(uavLabel(u))} · ${u.position[2].toFixed(1)} m · ${Math.round(u.battery_ratio*100)}%`);}
  const size=frame.heatmap.length,canvas=document.createElement('canvas');canvas.width=size;canvas.height=size;const ctx=canvas.getContext('2d'),pixels=ctx.createImageData(size,size);for(let row=0;row<size;row++)for(let col=0;col<size;col++){const value=frame.heatmap[row][col],offset=((size-1-row)*size+col)*4;pixels.data[offset]=235;pixels.data[offset+1]=165;pixels.data[offset+2]=65;pixels.data[offset+3]=Math.round(Math.min(1,value*4)*170*frame.heatmap_mask[row][col]);}ctx.putImageData(pixels,0,0);if(heatLayer)heatLayer.remove();heatLayer=L.imageOverlay(canvas.toDataURL(),frame.bounds_wgs84.map(p=>[p[1],p[0]]),{opacity:.8,interactive:false}).addTo(map);polygonLayer.bringToFront();}
async function validate(){if(!await checkFields())throw Error('Correct the highlighted fields.');const result=await api('/api/validate',config);setMode('idle');applyFrame(result.frame,true);return result;}
function pause(){playing=false;playGeneration++;$('play-simulation').textContent='Play';}
async function reset(stopPlay=true){if(!await checkFields())throw Error('Correct the highlighted fields.');if(stopPlay)pause();setMode('idle');const frame=await api('/api/simulation/reset',config);simulationReady=true;applyFrame(frame,true);}
async function step(){if(!simulationReady)await reset(false);const frame=await api('/api/simulation/step',{policy:$('simulation-policy').value});applyFrame(frame);}
async function play(){if(playing){pause();return;}playing=true;const generation=++playGeneration;$('play-simulation').textContent='Pause';while(playing&&generation===playGeneration){try{await step();}catch(e){notice(e.message,true);pause();}if(playing&&generation===playGeneration)await new Promise(r=>setTimeout(r,120));}}
function diagnostics(metrics){if(!metrics)return;const fields=[['Policy loss','actor_loss'],['Critic loss','value_loss'],['Gaussian entropy','entropy'],['Approximate KL','approximate_kl'],['Explained variance','explained_variance'],['Coverage actor fraction','actor_sample_fraction'],['Energy · Wh/decision','energy_Wh_per_step'],['Projection fraction','reward_projection'],['Collision fraction','reward_collision'],['Crashed fraction','reward_crash']];$('training-diagnostics').innerHTML='<dl>'+fields.map(([label,key])=>`<div><dt>${label}</dt><dd>${Number(metrics[key]).toFixed(3)}</dd></div>`).join('')+'</dl>';}
function chart(id,history,key,color){const canvas=$(id),ratio=window.devicePixelRatio||1;canvas.width=canvas.clientWidth*ratio;canvas.height=170*ratio;const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);const w=canvas.clientWidth,h=170;ctx.clearRect(0,0,w,h);if(!history.length)return;const values=history.map(x=>x[key]),min=Math.min(0,...values),max=Math.max(.01,...values);ctx.strokeStyle='#344c5e';ctx.fillStyle='#9ab0c0';ctx.font='11px Segoe UI';for(let i=0;i<3;i++){const y=15+i*58;ctx.beginPath();ctx.moveTo(45,y);ctx.lineTo(w-8,y);ctx.stroke();ctx.fillText((max-(max-min)*i/2).toFixed(2),3,y+3);}ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();history.forEach((point,index)=>{const x=45+(w-58)*index/Math.max(1,history.length-1),y=15+116*(1-(point[key]-min)/(max-min||1));index?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();ctx.fillText('0',45,160);ctx.fillText(history.at(-1).steps.toLocaleString()+' steps',Math.max(45,w-100),160);}
async function poll(){try{const job=await api('/api/training/status');activeJob=['starting','running','stopping'].includes(job.status);$('job-badge').textContent=job.status==='idle'?'Ready':job.status[0].toUpperCase()+job.status.slice(1);$('connection-status').textContent='Local backend connected';$('start-training').disabled=activeJob;$('stop-training').disabled=!activeJob;if(job.total_steps){$('progress-fill').style.width=Math.min(100,100*job.steps/job.total_steps)+'%';$('progress-text').textContent=`${(job.steps||0).toLocaleString()} / ${job.total_steps.toLocaleString()} steps${job.metrics?' · '+job.metrics.steps_per_second.toFixed(1)+' steps/s':''}`;}if(job.frame&&currentTab==='training')applyFrame(job.frame);diagnostics(job.metrics);chart('coverage-chart',job.history||[],'mean_coverage','#e9b36c');chart('reward-chart',job.history||[],'mean_step_reward','#8cb9cd');if(job.status==='failed'&&job.error!==poll.lastError){notice(job.error,true);poll.lastError=job.error;}}catch(e){$('connection-status').textContent='Backend disconnected';}setTimeout(poll,1000);}
async function loadRuns(){const runs=await api('/api/runs');$('run-list').innerHTML='';if(!runs.length){$('run-list').innerHTML='<p class="muted">No runs yet. Start an experiment in Training.</p>';return;}for(const run of runs){const item=document.createElement('div');item.className='run-block';const evaluation=run.evaluation;item.innerHTML=`<h3>${escapeHTML(run.run_id)}</h3><p>${escapeHTML(run.status)} · ${run.steps.toLocaleString()} steps · seed ${run.seed}<br>${evaluation?'Last evaluation return '+evaluation.return_mean.toFixed(2)+' · mean quality '+evaluation.coverage_mean.toFixed(3):'No scheduled evaluation recorded.'}</p><label>Checkpoint<select>${run.checkpoints.map(c=>`<option ${c==='best.pt'?'selected':''}>${escapeHTML(c)}</option>`).join('')}</select></label><div class="toolbar"><button class="replay">Replay</button><button class="evaluate">Evaluate</button><a href="/api/runs/${encodeURIComponent(run.run_id)}/export">Export ZIP</a></div>`;item.querySelector('.replay').onclick=async()=>{try{const result=await api(`/api/runs/${encodeURIComponent(run.run_id)}/load`,{checkpoint:item.querySelector('select').value});pause();setMode('idle');config=result.config;normalizeMission(config.scenario,result.frame);dockPoints=result.frame.stations.map(s=>({type:s.type_index,geographic:s.geographic}));renderForms();simulationReady=true;applyFrame(result.frame,true);$('simulation-policy').value='model';setTab('mission');notice('Saved policy loaded. Use Step or Play to replay.');}catch(e){notice(e.message,true);}};item.querySelector('.evaluate').onclick=async(event)=>{event.target.disabled=true;try{const result=await api(`/api/runs/${encodeURIComponent(run.run_id)}/evaluate`,{checkpoint:item.querySelector('select').value,episodes:config.training.evaluation_episodes});applyFrame(result.frame,true);notice(`Evaluation: return ${result.return_mean.toFixed(2)} · mean quality ${result.coverage_mean.toFixed(3)} · crashes ${result.crashed_uavs.toFixed(1)}`);}catch(e){notice(e.message,true);}finally{event.target.disabled=false;}};$('run-list').append(item);}}

async function initialize() {
  try {
    const response=await api('/api/defaults');defaults=response.config;schema=response.schema;presets=response.presets;config=structuredClone(defaults);normalizeMission(config.scenario);renderForms();
    if(window.L) {
      map=L.map('map',{doubleClickZoom:false}).setView([40.63,22.94],15);
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
          config.scenario.polygon_wgs84=[...draft];dockPoints=[];previewLayer?.remove();syncDocks();simulationReady=false;setMode('dock');
        }
      });
    } else {$('map').hidden=true;$('map-fallback').hidden=false;}
    const initial=await api('/api/validate',config);applyFrame(initial.frame,true);
    dockPoints=initial.frame.stations.map(s=>({type:0,geographic:s.geographic}));
    document.querySelectorAll('nav button').forEach(button=>button.onclick=()=>setTab(button.dataset.tab));
    action('validate',validate);action('reset-simulation',()=>reset());action('step-simulation',step);$('play-simulation').onclick=play;
    action('draw-area',()=>{pause();draft=[];simulationReady=false;setMode('draw');polygonLayer?.remove();heatLayer?.remove();});
    action('place-station',()=>setMode(mode==='dock'?'idle':'dock'));
    action('clear-docks',()=>{dockPoints=[];syncDocks();});
    action('add-uav',()=>{
      config.scenario.uav_types.push({...structuredClone(presets.uavs[0].values),name:'UAV '+(config.scenario.uav_types.length+1)});
      config.scenario.uav_launch_docks.push(0);renderUAVs();changed();
    });
    action('start-training',async()=>{pause();await validate();await api('/api/training/start',config);});
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
    poll();
  } catch(error){notice(error.message,true);$('connection-status').textContent='Unable to initialize';}
}
initialize();
