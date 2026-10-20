'use strict';
const fieldHelp = {
  required_gsd_m:'Maximum acceptable ground size represented by one image pixel. A cell captured above this limit earns no quality. Example: 5 cm/pixel.',
  delta_t:'Simulated time advanced by one decision. It affects movement, energy and coverage ageing in both training and simulation. Example: 500 steps × 5 s = 2,500 simulated seconds.',
  horizon:'Maximum decisions in each training or evaluation episode. The mission resets at this limit, or earlier if all UAVs crash. Example: 500 steps at ΔT = 5 s lasts 41 min 40 s.',
  min_altitude_m:'Lowest permitted altitude during coverage. Automatic takeoff and landing can pass below it.',
  max_altitude_m:'Highest permitted coverage altitude above the flat ground plane.',
  coverage_lambda:'How much previous coverage quality survives each step. With λ = 0.9, 90% is retained and 10% comes from current captures.',
  grid_size:'Cells along each side of the detailed coverage grid. Example: 1,024 means up to 1,024² cells; a finer grid costs more computation.',
  observation_size:'Cells along each side of the map supplied to the policy. Each block summarizes detailed coverage. It must divide the coverage-grid size.',
  name:'Your name for this UAV. Its numbered identifier remains visible on the map.',
  preset:'Fill properties from a reference model. These are simulator templates; the energy model includes assumptions that may need calibration.',
  launch_dock:'Dock from which this UAV starts. Several UAVs can share a dock and take off in sequence.',
  fixed_count:'Number of charging docks in this mission. Place the same number on the map.',
  max_horizontal_speed_mps:'Maximum horizontal flight speed. The acceleration limits also restrict how far the UAV can move in one step.',
  max_climb_speed_mps:'Maximum upward speed.', max_descent_speed_mps:'Maximum downward speed.',
  max_horizontal_accel_mps2:'Maximum horizontal acceleration and braking. Example: 4 m/s² changes speed by up to 4 m/s each second.',
  max_vertical_accel_mps2:'Maximum vertical acceleration and braking.',
  max_yaw_rate_degps:'Maximum rotational speed of the UAV heading. Example: 90°/s permits a quarter turn per second before acceleration limits.',
  max_yaw_accel_degps2:'Maximum rate of change of heading rotational speed.',
  max_gimbal_pitch_rate_degps:'Maximum speed at which the camera tilts up or down.',
  gimbal_pitch_min_deg:'Lowest camera tilt. −90° points straight down; 0° points toward the horizon.',
  gimbal_pitch_max_deg:'Highest allowed camera tilt. Upward-looking views do not capture the ground.',
  battery_capacity_Wh:'Total battery energy. Every UAV starts at 100%. Example: 77 Wh can ideally supply 77 W for one hour.',
  reserve_battery_ratio:'Battery fraction reserved when the scheduler estimates a safe return. Example: 15% is a return reserve, not the initial charge.',
  P_airborne_base_W:'Baseline airborne power, including hovering. It is part of the empirical energy model, not a PID setting.',
  k_vxy:'Energy-model coefficient for horizontal speed squared. Larger values make fast horizontal movement more expensive.',
  k_vz_up:'Additional power coefficient for upward speed.', k_vz_down:'Additional power coefficient for downward speed.',
  k_acc:'Energy-model coefficient for acceleration squared. Larger values penalize changes in velocity more strongly.',
  takeoff_energy_Wh:'Extra energy charged once for each automatic takeoff.',
  landing_energy_Wh:'Extra energy charged for the final automatic landing step.',
  collision_radius_m:'Effective UAV radius for collision checks and landing approach. Two UAVs collide when their separation falls below the sum of their radii.',
  camera_width_px:'Horizontal image resolution used in camera geometry and ground-resolution calculations.',
  camera_height_px:'Vertical image resolution used in camera geometry and ground-resolution calculations.',
  camera_fov_deg:'Camera field of view along the selected FOV axis. A wider view covers more ground but may reduce ground resolution.',
  camera_fov_axis:'Whether the stated field of view spans the image diagonal or its width.',
  charge_power_W:'Constant effective power delivered to the UAV battery. Docks have unlimited external supply and charge one UAV at a time.',
  total_steps:'Total mission steps collected for training in the single environment. Each step advances all UAVs together; their individual transitions remain separate. Episode resets do not reset this training counter.',
  rollout_steps:'Mission steps collected before an IPPO update phase. Example: every 100 steps, update using that buffer and then collect fresh experience. Each phase can contain several minibatches and epochs.',
  evaluation_frequency:'Total team steps between policy evaluations, checked after each rollout. Evaluation does not change the weights. Example: an interval crossed mid-rollout runs at the next update boundary.',
  evaluation_episodes:'Complete episodes used for each evaluation. Each uses Episode length and may end earlier if all UAVs crash. Their average return determines the best checkpoint.',
  checkpoint_frequency:'Total team steps between saved model checkpoints, checked after a rollout update. Checkpoints include weights, optimizer state and configuration. Initial and final models are also saved.',
  seed:'Seed for repeatable initialization and action sampling. Evaluation uses separate fixed seeds.',
  torch_threads:'CPU threads used by PyTorch for neural-network calculations, out of the detected total. The single simulator also does CPU work. This is not a cap on total application CPU usage.',
  device:'Auto uses a compatible GPU if its estimated free memory is sufficient, otherwise CPU. Simulation stays on CPU. GPU detection does not install or change drivers or libraries.',
  learning_rate:'Size of each optimizer update. Larger values learn faster but can destabilize the policy. Example: 0.0003.',
  gamma:'How strongly future rewards count. Values near 1 favour longer-term outcomes. Example: 0.99 discounts a reward one step ahead by 1%.',
  gae_lambda:'How much advantage estimation relies on longer sequences of rewards rather than the value prediction. Higher values use longer sequences but can be noisier.',
  clip_range:'Limits the importance-ratio change used by the PPO policy objective. Example: 0.2 gives a clipping range of 0.8–1.2.',
  entropy_coef:'Weight encouraging action exploration. Higher values favour more varied actions.',
  value_coef:'Weight of value-prediction error relative to policy loss in each joint optimizer update.',
  max_grad_norm:'Maximum gradient norm before an optimizer step. Large gradients are scaled down to this limit.',
  batch_size:'Team frames processed per optimizer step, including their separate valid UAV samples. Example: 100 collected frames with a minibatch size of 64 gives two minibatches: 64 and 36.',
  epochs:'Complete passes over the collected update buffer. Each pass shuffles the frames and updates on every minibatch. Example: 4 epochs × 2 minibatches = 8 optimizer steps, unless Target KL stops earlier.',
  hidden_size:'Shared sizing parameter H, in neurons. Own-state and map embeddings have width H. Each teammate is encoded to H/2, then mean/max pooling produces H features. Actor and critic each have two hidden layers of H neurons. CNN channels stay fixed at 16 and 24. H must be even.',
  target_kl:'Approximate limit on how far the updated policy moves from the rollout policy. Crossing it stops the remaining update passes. It is a stability threshold, not a target reward.',
  mean_weight:'Weight of the mean-quality difference caused by this UAV’s capture. The three coverage weights must sum to 1.',
  fairness_weight:'Weight of the Jain-fairness difference caused by this UAV’s capture. It can be negative even when some coverage improves.',
  p10_weight:'Weight of the change in the 10th-percentile cell quality. It favours improvement in poorly covered regions.',
  energy_weight:'Penalty weight for this UAV’s normalized energy use, including automatic flight phases.',
  projection_weight:'Penalty weight for correction of this UAV’s proposed action to its feasible flight envelope.',
  collision_weight:'Penalty weight for a collision involving this UAV. Uninvolved UAVs receive no collision penalty.',
  crash_weight:'Persistent penalty weight for this UAV while crashed, until the episode ends.',
  place_search:'Search for a place and move the map there. Searching does not replace the mission polygon.',
  simulation_policy:'Mission preview only. Sweep baseline follows hand-written cyclic targets with a downward camera; it is a simple comparison policy, not a learned or optimized coverage plan. Random actions samples the allowed action space. Both still use the flight envelope and recharge scheduler. Training always uses the shared IPPO policy.',
  mean_quality:'Average current coverage quality across cells inside the area, from 0 to 1. Example: 0.438 means a mean cell quality of 0.438, not 43.8% of the area visited. Quality decays between captures and starts at zero each episode.',
  p10_quality:'10th-percentile coverage quality inside the area, from 0 to 1. About 10% of cells have quality at or below this value. It is not the average of the worst 10%. Starts at zero each episode.',
  spatial_fairness:'Jain’s index across coverage cells: (sum of quality)² / (number of cells × sum of squared quality). Near 1 means uniform quality; lower values mean uneven coverage. All-zero coverage is reported as 0. Uniform weak coverage can still score near 1. Resets each episode.',
  live_reward:'Reward for the latest single step, averaged across actual UAVs. Each UAV learns from its own reward; this average is only for monitoring. It is not a cumulative return or an evaluation result.',
  simulation_time:'Simulated time within the displayed episode: episode step × ΔT. It resets when a new episode begins. Wall-clock processing time is separate.',
  episode_counter:'Current training episode and completed step within it. Coverage, UAV batteries and simulated episode time reset at the next episode; total training steps continue. An episode may end early if all UAVs crash.',
  coverage_plot:'Current mean cell quality at every training step, independent of weight updates and evaluation. Lines break at episode boundaries. The latest 2,000 points are shown; steps.csv retains every point.',
  reward_plot:'Individual UAV rewards averaged for each training step, without rollout averaging or accumulation. Lines break at episode boundaries. The latest 2,000 points are shown; full per-UAV readings are saved in uav-rewards.csv.',
  fleet_reading:'Latest-step battery percentage, flight status and individual reward. UAVs start each new episode at 100%. Charging may also increase the battery during an episode.',
  safety_reading:'Fleet energy consumed during the latest displayed step, number of currently crashed UAVs, and collision pairs detected in that step. These are single-step readings, not episode averages.',
  phase_reading:'Current training phase. Training steps pause during weight updates, scheduled evaluation and checkpoint saving. Evaluation has separate episodes and does not enter the training plots.',
  checkpoint:'Saved weights to replay or evaluate. best.pt has the highest mean evaluation return; final.pt contains the latest weights.',
  actor_loss:'Objective used to improve the shared policy, averaged over executed UAV actions.',
  value_loss:'Error in predicted individual UAV returns. Lower values mean closer predictions for this batch.',
  entropy:'Uncertainty of the underlying Gaussian action distribution; a diagnostic of exploration.',
  approximate_kl:'Measured policy change during the update. Target KL stops further passes when this grows too large.',
  explained_variance:'How well value predictions explain variation in returns. Near 1 is better; near 0 predicts little of the variation.',
  actor_sample_fraction:'Fraction of actual UAV transitions in which the policy action executed. Automatic flight and charging phases are excluded from policy loss.',
  energy_Wh_per_step:'Mean fleet energy consumed per environment decision during this rollout.',
  reward_projection:'Mean normalized action-correction penalty before applying its weight.',
  reward_collision:'Fraction of UAV transitions involving a collision in this rollout.',
  reward_crash:'Fraction of UAV transitions spent crashed in this rollout.'
};

function attachFieldHelp(element,key,extra='') {
  const description=fieldHelp[key];
  if(description)element.dataset.help=description+(extra?' '+extra:'');
  return element;
}

function initializeFieldHelp() {
  const tooltip=document.createElement('div');tooltip.id='field-help';tooltip.role='tooltip';tooltip.hidden=true;
  document.body.append(tooltip);
  let timer,anchor,control;
  function hide() {
    clearTimeout(timer);tooltip.hidden=true;
    if(control) {
      const ids=(control.getAttribute('aria-describedby')||'').split(' ').filter(id=>id&&id!==tooltip.id);
      if(ids.length)control.setAttribute('aria-describedby',ids.join(' '));else control.removeAttribute('aria-describedby');
    }
    anchor=null;control=null;
  }
  function schedule(target) {
    if(target===anchor)return;
    hide();if(!target)return;anchor=target;
    timer=setTimeout(()=>{
      if(!anchor?.isConnected){hide();return;}
      tooltip.textContent=anchor.dataset.help;tooltip.hidden=false;
      const box=anchor.getBoundingClientRect(),width=tooltip.offsetWidth,height=tooltip.offsetHeight;
      const left=box.right+12+width<innerWidth?box.right+12:Math.max(12,Math.min(box.left,innerWidth-width-12));
      tooltip.style.left=left+'px';tooltip.style.top=Math.max(12,Math.min(box.top,innerHeight-height-12))+'px';
      control=anchor.matches('input,select')?anchor:anchor.querySelector('input,select');
      if(control)control.setAttribute('aria-describedby',[control.getAttribute('aria-describedby'),tooltip.id].filter(Boolean).join(' '));
    },1500);
  }
  document.addEventListener('pointerover',event=>schedule(event.target.closest('[data-help]')));
  document.addEventListener('pointerout',event=>{
    if(anchor&&!anchor.contains(event.relatedTarget))hide();
  });
  document.addEventListener('focusin',event=>schedule(event.target.closest('[data-help]')));
  document.addEventListener('focusout',event=>{if(anchor&&!anchor.contains(event.relatedTarget))hide();});
  document.addEventListener('pointerdown',hide);
  document.addEventListener('input',hide);
  document.addEventListener('scroll',hide,true);
  document.addEventListener('keydown',event=>{if(event.key==='Escape')hide();});
  window.addEventListener('blur',hide);
}
