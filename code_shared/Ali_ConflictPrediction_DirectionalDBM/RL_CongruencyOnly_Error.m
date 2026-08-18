%% Fit congruency-only conflict-learning model to errors
%
% Conflict is defined only by actual congruency:
%   conflict(t) = 0 for congruent and 1 for incongruent trials.
% Distance remains a nuisance regressor in the error GLM but does not
% update the learned expected-conflict state.

clear
clc

rootDir = fileparts(mfilename('fullpath'));
settings = struct;
settings.nSubjectsRequested = Inf;
settings.resetAtBlock = true;
settings.resume = true;
settings.useAlphaPrior = true;
settings.ridgePenalty = 1e-4;
settings.conflictDefinition = 'congruencyOnly';
settings.modelStem = 'CongruencyOnlyRL';
settings.outputRoot = fullfile(rootDir, 'CC_RL_Models', ...
    'CongruencyOnlyRL');

fitConflictHistoryRL('Error', settings);
