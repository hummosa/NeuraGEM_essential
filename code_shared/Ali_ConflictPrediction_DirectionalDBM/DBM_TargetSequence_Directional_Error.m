%% Fit the direction-specific target-sequence DBM to errors
%
% This is the Markov-order-1 extension of DBM_TargetSequence_Error.m. It
% maintains separate beliefs about P(right | previous left) and
% P(right | previous right), while retaining the same logistic error
% regression, exclusions, alpha search, and weak ridge stabilization.

clear
clc

rootDir = fileparts(mfilename('fullpath'));
if isempty(rootDir)
    rootDir = pwd;
end

settings = struct;
settings.nSubjectsRequested = Inf;
settings.resume = true;

fitTargetSequenceDirectionalDBM('Error', rootDir, settings);

