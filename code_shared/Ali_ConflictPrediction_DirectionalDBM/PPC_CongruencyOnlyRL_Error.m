%% Conflict-history PPC for the congruency-only error model

clear
clc

settings = struct;
settings.modelStem = 'CongruencyOnlyRL';
settings.nReplications = 500;
settings.nSubjectsRequested = Inf;
settings.figureVisible = 'off';

runConflictHistoryRLPPC('Error', settings);
