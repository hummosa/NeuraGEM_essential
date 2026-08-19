%% Fit conflict-prediction reinforcement learning model to errors
%
% The trialwise expected-conflict state is learned with the delta rule in
% RL_modeling_CongruencyOnly.m. Error probability is predicted from
% expected conflict, current incongruence, their interaction, and nuisance
% terms. A very weak ridge penalty stabilizes rare participant-level
% separation without penalizing the intercept.

clear
clc

settings = struct;
settings.nSubjectsRequested = Inf;
settings.resetAtBlock = true;
settings.resume = true;
settings.ridgePenalty = 1e-4;
settings.useAlphaPrior = true;

fitConflictHistoryRL('Error', settings);
