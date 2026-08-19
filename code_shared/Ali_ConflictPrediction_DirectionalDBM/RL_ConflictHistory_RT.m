%% Fit conflict-prediction reinforcement learning model to reaction time
%
% The trialwise expected-conflict state is learned with the delta rule in
% RL_modeling_CongruencyOnly.m. The observation model predicts log RT on
% correct trials from expected conflict, current incongruence, their
% interaction, and the same kinds of nuisance terms used in the DBM
% pipeline.

clear
clc

settings = struct;
settings.nSubjectsRequested = Inf;
settings.resetAtBlock = true;
settings.resume = true;
settings.useAlphaPrior = true;

fitConflictHistoryRL('RT', settings);
