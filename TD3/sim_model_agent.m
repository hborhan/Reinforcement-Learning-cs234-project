clc
clear all
bdclose all

% Specify the initial position and velocity for the two vehicles.
x0_lead = 38;   % initial position for lead car (m)
v0_lead = 29;   % initial velocity for lead car (m/s)
x0_ego = 0;    % initial position for ego car (m)
v0_ego = 29;    % initial velocity for ego car (m/s)

x0_lead = 10;   % initial position for lead car (m)
v0_lead = 29;   % initial velocity for lead car (m/s)
x0_ego = 0;    % initial position for ego car (m)
v0_ego = 29;    % initial velocity for ego car (m/s)


% Specify standstill default spacing (m), time gap (s) and driver-set velocity (m/s).
D_default = 10;
t_gap = 0.6;
d_target = 17;
v_set = 30;

% Considering the physical limitations of the vehicle dynamics, the acceleration is constrained to the range [-3,2] (m/s^2).
amin_ego = -2;
amax_ego = 1;

% Define the sample time Ts and simulation duration Tf in seconds.
Ts = 0.1;
Tf = 80;

load('D:\Stanford\CS221\project\Final\TD3\run1\Agents\Agent650.mat', 'saved_agent')

agent=saved_agent

mdl = 'ModelR3_sim';
open_system(mdl)
sim(mdl)
