clc
clear all
bdclose all

%%
x0_lead = 38;   % initial position for lead car (m)
v0_lead = 29;   % initial velocity for lead car (m/s)
x0_ego = 0;    % initial position for ego car (m)
v0_ego = 29;    % initial velocity for ego car (m/s)

%%
D_default = 10;
t_gap = 0.6;
d_target = 17;
v_set = 30;

%%
amin_ego = -2;
amax_ego = 1;

%%
Ts = 0.1;
Tf = 60;

%%
mdl = 'ModelR3';
open_system(mdl)
agentblk = [mdl '/RL Agent'];

%%
% create the observation info
observationInfo = rlNumericSpec([4 1],'LowerLimit',-inf*ones(4,1),'UpperLimit',inf*ones(4,1));
observationInfo.Name = 'observations';
observationInfo.Description = 'states';
% action Info
% actionInfo = rlNumericSpec([1 1],'LowerLimit',-3,'UpperLimit',2);
actionInfo = rlNumericSpec([1 1],'LowerLimit',amin_ego,'UpperLimit',amax_ego);
actionInfo.Name = 'acceleration';
% define environment
env = rlSimulinkEnv(mdl,agentblk,observationInfo,actionInfo);

numObs = 4;
numAct = 1;
%%
% randomize initial positions of lead car
env.ResetFcn = @(in)localResetFcnCACC(in);

%%
rng('default')

%% create critic
criticLayerSizes = [400 300];

%% First Critic network
statePath1 = [
    imageInputLayer([numObs 1 1],'Normalization','none','Name', 'observation')
    fullyConnectedLayer(criticLayerSizes(1), 'Name', 'CriticStateFC1', ... 
            'Weights',2/sqrt(numObs)*(rand(criticLayerSizes(1),numObs)-0.5), ...
            'Bias',2/sqrt(numObs)*(rand(criticLayerSizes(1),1)-0.5))
    reluLayer('Name','CriticStateRelu1')
    fullyConnectedLayer(criticLayerSizes(2), 'Name', 'CriticStateFC2', ...
            'Weights',2/sqrt(criticLayerSizes(1))*(rand(criticLayerSizes(2),criticLayerSizes(1))-0.5), ... 
            'Bias',2/sqrt(criticLayerSizes(1))*(rand(criticLayerSizes(2),1)-0.5))
    ];
actionPath1 = [
    imageInputLayer([numAct 1 1],'Normalization','none', 'Name', 'action')
    fullyConnectedLayer(criticLayerSizes(2), 'Name', 'CriticActionFC1', ...
            'Weights',2/sqrt(numAct)*(rand(criticLayerSizes(2),numAct)-0.5), ... 
            'Bias',2/sqrt(numAct)*(rand(criticLayerSizes(2),1)-0.5))
    ];
commonPath1 = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu1')
    fullyConnectedLayer(1, 'Name', 'CriticOutput',...
            'Weights',2*5e-3*(rand(1,criticLayerSizes(2))-0.5), ...
            'Bias',2*5e-3*(rand(1,1)-0.5))
    ];

% Connect the layer graph
criticNetwork1 = layerGraph(statePath1);
criticNetwork1 = addLayers(criticNetwork1, actionPath1);
criticNetwork1 = addLayers(criticNetwork1, commonPath1);
criticNetwork1 = connectLayers(criticNetwork1,'CriticStateFC2','add/in1');
criticNetwork1 = connectLayers(criticNetwork1,'CriticActionFC1','add/in2');

%% Second Critic network 
statePath2 = [
    imageInputLayer([numObs 1 1],'Normalization','none','Name', 'observation')
    fullyConnectedLayer(criticLayerSizes(1), 'Name', 'CriticStateFC1', ... 
            'Weights',2/sqrt(numObs)*(rand(criticLayerSizes(1),numObs)-0.5), ...
            'Bias',2/sqrt(numObs)*(rand(criticLayerSizes(1),1)-0.5))
    reluLayer('Name','CriticStateRelu1')
    fullyConnectedLayer(criticLayerSizes(2), 'Name', 'CriticStateFC2', ...
            'Weights',2/sqrt(criticLayerSizes(1))*(rand(criticLayerSizes(2),criticLayerSizes(1))-0.5), ... 
            'Bias',2/sqrt(criticLayerSizes(1))*(rand(criticLayerSizes(2),1)-0.5))
    ];
actionPath2 = [
    imageInputLayer([numAct 1 1],'Normalization','none', 'Name', 'action')
    fullyConnectedLayer(criticLayerSizes(2), 'Name', 'CriticActionFC1', ...
            'Weights',2/sqrt(numAct)*(rand(criticLayerSizes(2),numAct)-0.5), ... 
            'Bias',2/sqrt(numAct)*(rand(criticLayerSizes(2),1)-0.5))
    ];
commonPath2 = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu1')
    fullyConnectedLayer(1, 'Name', 'CriticOutput',...
            'Weights',2*5e-3*(rand(1,criticLayerSizes(2))-0.5), ...
            'Bias',2*5e-3*(rand(1,1)-0.5))
    ];

% Connect the layer graph
criticNetwork2 = layerGraph(statePath2);
criticNetwork2 = addLayers(criticNetwork2, actionPath2);
criticNetwork2 = addLayers(criticNetwork2, commonPath2);
criticNetwork2 = connectLayers(criticNetwork2,'CriticStateFC2','add/in1');
criticNetwork2 = connectLayers(criticNetwork2,'CriticActionFC1','add/in2');


%%
% criticOptions = rlRepresentationOptions('LearnRate',1e-3,'GradientThreshold',1,'L2RegularizationFactor',1e-4);
criticOptions = rlRepresentationOptions('Optimizer','adam','LearnRate',1e-3,... 
                                        'GradientThreshold',1,'L2RegularizationFactor',2e-4);
critic1 = rlQValueRepresentation(criticNetwork1,observationInfo,actionInfo,...
    'Observation',{'observation'},'Action',{'action'},criticOptions);
critic2 = rlQValueRepresentation(criticNetwork2,observationInfo,actionInfo,...
    'Observation',{'observation'},'Action',{'action'},criticOptions);

%% create actor
actorLayerSizes = [400 300];
actorNetwork = [
    imageInputLayer([numObs 1 1],'Normalization','none','Name','observation')
    fullyConnectedLayer(actorLayerSizes(1), 'Name', 'ActorFC1', ...
            'Weights',2/sqrt(numObs)*(rand(actorLayerSizes(1),numObs)-0.5), ... 
            'Bias',2/sqrt(numObs)*(rand(actorLayerSizes(1),1)-0.5))
    reluLayer('Name', 'ActorRelu1')
    fullyConnectedLayer(actorLayerSizes(2), 'Name', 'ActorFC2', ... 
            'Weights',2/sqrt(actorLayerSizes(1))*(rand(actorLayerSizes(2),actorLayerSizes(1))-0.5), ... 
            'Bias',2/sqrt(actorLayerSizes(1))*(rand(actorLayerSizes(2),1)-0.5))
    reluLayer('Name', 'ActorRelu2')
    fullyConnectedLayer(numAct, 'Name', 'ActorFC3', ... 
                       'Weights',2*5e-3*(rand(numAct,actorLayerSizes(2))-0.5), ... 
                       'Bias',2*5e-3*(rand(numAct,1)-0.5))                       
    tanhLayer('Name','ActorTanh1')
    scalingLayer('Name','ActorScaling1','Scale',1.5,'Bias',-0.5)
    ];

%%
% actorOptions = rlRepresentationOptions('LearnRate',1e-4,'GradientThreshold',1,'L2RegularizationFactor',1e-4);
actorOptions = rlRepresentationOptions('Optimizer','adam','LearnRate',1e-3,...
                                       'GradientThreshold',1,'L2RegularizationFactor',1e-5);
actor = rlDeterministicActorRepresentation(actorNetwork,observationInfo,actionInfo,...
    'Observation',{'observation'},'Action',{'ActorScaling1'},actorOptions);


%% Specify TD3 agent options
agentOptions = rlTD3AgentOptions;
agentOptions.SampleTime = Ts;
agentOptions.DiscountFactor = 0.99;
agentOptions.MiniBatchSize = 256;
agentOptions.ExperienceBufferLength = 1e6;
agentOptions.TargetSmoothFactor = 5e-3;
agentOptions.TargetPolicySmoothModel.Variance = 0.2; % target policy noise 0.2
agentOptions.TargetPolicySmoothModel.LowerLimit = -0.5; % -0.5
agentOptions.TargetPolicySmoothModel.UpperLimit = 0.5; % 0.5
agentOptions.ExplorationModel = rl.option.OrnsteinUhlenbeckActionNoise; % set up OU noise as exploration noise (default is Gaussian for rlTD3AgentOptions)
agentOptions.ExplorationModel.MeanAttractionConstant = 1; % 1
agentOptions.ExplorationModel.Variance = 0.1; % 0.1
% agentOptions.ExplorationModel.MeanAttractionConstant = 0.5;
% agentOptions.ExplorationModel.Variance = 1;


agent = rlTD3Agent(actor,[critic1 critic2],agentOptions);

% delayedDDPGAgent = rlTD3Agent(actor,critic1,agentOptions);

%% train agent
maxepisodes = 655;
maxsteps = ceil(Tf/Ts);
trainingOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes,...
    'MaxStepsPerEpisode',maxsteps,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','EpisodeCount',...
    'StopTrainingValue',maxepisodes,...
    'SaveAgentCriteria','EpisodeCount',...
    'SaveAgentValue',1,...
    'SaveAgentDirectory', pwd + "\run1\Agents",...
    'Plots',"training-progress");
trainOpts.UseParallel = true;
trainOpts.ParallelizationOptions.Mode = 'async';
trainOpts.ParallelizationOptions.StepsUntilDataIsSent = 32;
trainOpts.ParallelizationOptions.DataToSendFromWorkers = 'Experiences';

%%

trainingStats = train(agent,env,trainingOpts);
%trainingStats = train(delayedDDPGAgent,env,trainingOpts);


