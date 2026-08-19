%% Parametric predictive checks for conflict-history effects on errors

clear
clc

settings = struct;
settings.nReplications = 500;
settings.nSubjectsRequested = Inf;
settings.figureVisible = 'off';

runConflictHistoryRLPPC('Error', settings);
