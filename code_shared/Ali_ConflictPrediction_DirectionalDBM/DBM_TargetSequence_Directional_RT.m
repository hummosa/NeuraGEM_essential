%% Fit the direction-specific target-sequence DBM to reaction time
%
% This is the Markov-order-1 extension of DBM_TargetSequence_RT.m. It
% maintains separate beliefs about P(right | previous left) and
% P(right | previous right), while retaining the same RT regression,
% exclusions, alpha search, and block resets as the original model.

clear
clc

rootDir = fileparts(mfilename('fullpath'));
if isempty(rootDir)
    rootDir = pwd;
end

settings = struct;
settings.nSubjectsRequested = Inf;
settings.resume = true;

fitTargetSequenceDirectionalDBM('RT', rootDir, settings);

