%% Parametric predictive checks for the directional target DBM RT model

clear
clc

rootDir = fileparts(mfilename('fullpath'));
if isempty(rootDir)
    rootDir = pwd;
end

ppcConfig = struct;
ppcConfig.fitDir = fullfile(rootDir, 'CC_RL_Models', ...
    'TargetSequenceDirectionalDBM', 'RT', 'SubjectData');
ppcConfig.outputDir = fullfile(rootDir, 'Figures', ...
    'TargetSequenceDirectionalDBM', 'RT_PPC');
ppcConfig.fitPattern = '*_TargetSequenceDirectionalDBM_RT.mat';
ppcConfig.outputStem = 'PPC_TargetSequenceDirectionalDBM_RT';

run(fullfile(rootDir, 'PPC_TargetSequenceDBM_RT.m'))

