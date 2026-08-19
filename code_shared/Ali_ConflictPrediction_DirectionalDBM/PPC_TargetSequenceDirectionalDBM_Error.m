%% Parametric predictive checks for the directional target DBM error model

clear
clc

rootDir = fileparts(mfilename('fullpath'));
if isempty(rootDir)
    rootDir = pwd;
end

ppcConfig = struct;
ppcConfig.fitDir = fullfile(rootDir, 'CC_RL_Models', ...
    'TargetSequenceDirectionalDBM', 'Error', 'SubjectData');
ppcConfig.outputDir = fullfile(rootDir, 'Figures', ...
    'TargetSequenceDirectionalDBM', 'Error_PPC');
ppcConfig.fitPattern = '*_TargetSequenceDirectionalDBM_Error.mat';
ppcConfig.outputStem = 'PPC_TargetSequenceDirectionalDBM_Error';

run(fullfile(rootDir, 'PPC_TargetSequenceDBM_Error.m'))

