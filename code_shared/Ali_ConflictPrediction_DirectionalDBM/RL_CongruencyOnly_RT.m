%% Fit congruency-only conflict-learning model to reaction time
%
% Conflict is defined only by actual congruency:
%   conflict(t) = 0 for congruent and 1 for incongruent trials.
% Distance remains a nuisance regressor in the RT GLM but does not update
% the learned expected-conflict state.

clear
clc

rootDir = fileparts(mfilename('fullpath'));
settings = struct;
settings.nSubjectsRequested = Inf;
settings.resetAtBlock = true;
settings.resume = true;
settings.useAlphaPrior = true;
settings.conflictDefinition = 'congruencyOnly';
settings.modelStem = 'CongruencyOnlyRL';
settings.outputRoot = fullfile(rootDir, 'CC_RL_Models', ...
    'CongruencyOnlyRL');

fitConflictHistoryRL('RT', settings);
