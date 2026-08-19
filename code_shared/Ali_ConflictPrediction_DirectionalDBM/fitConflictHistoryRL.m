function rlResult = fitConflictHistoryRL(outcomeType, settings)
%FITCONFLICTHISTORYRL Fit a conflict-prediction RL model participant-wise.
%
% The latent state follows the delta rule used by
% RL_modeling_CongruencyOnly.m:
%
%   expectedConflict(t + 1) = expectedConflict(t) + ...
%       alpha * (trialConflict(t) - expectedConflict(t))
%
% trialConflict is the mean of current incongruence and close stimulus
% distance. The value available before the current trial predicts RT or
% error probability, both alone and in interaction with current
% incongruence. Alpha is profiled on a coarse grid and then refined; the
% regression coefficients are refit at every candidate alpha.

if nargin < 1 || isempty(outcomeType)
    error('Specify outcomeType as ''RT'' or ''Error''.')
end
if nargin < 2 || isempty(settings)
    settings = struct;
end

outcomeType = validatestring(outcomeType, {'RT', 'Error'});
rootDir = fileparts(mfilename('fullpath'));
if isempty(rootDir)
    rootDir = pwd;
end

settings = applyDefaults(settings, rootDir);

dataDir = settings.dataDir;
assert(isfolder(dataDir), 'Behavioral data directory was not found: %s', dataDir)

outputDir = fullfile(settings.outputRoot, outcomeType);
subjectOutputDir = fullfile(outputDir, 'SubjectData');
if ~isfolder(subjectOutputDir), mkdir(subjectOutputDir); end

modelName = [settings.modelStem '_' outcomeType];
resultFile = fullfile(outputDir, [modelName '_results.mat']);
checkpointFile = fullfile(outputDir, [modelName '_results_incomplete.mat']);

files = dir(fullfile(dataDir, '*.mat'));
assert(~isempty(files), 'No subject .mat files were found in %s.', dataDir)
[~, fileOrder] = sort({files.name});
files = files(fileOrder);
files = files(1:min(settings.nSubjectsRequested, numel(files)));
nSubjects = numel(files);

if strcmp(outcomeType, 'RT')
    parameterNames = {'alpha', 'intc', 'bExpectedConflict', ...
        'bIncongruence', 'bExpectedConflictXIncongruence', ...
        'bDistance', 'bRSI', 'bTrial', 'bPreviousIncongruence', ...
        'bPreviousError', 'bPreviousLogRT', 'logSigma'};
else
    parameterNames = {'alpha', 'intc', 'bExpectedConflict', ...
        'bIncongruence', 'bExpectedConflictXIncongruence', ...
        'bDistance', 'bRSI', 'bTrial', 'bPreviousIncongruence', ...
        'bPreviousError'};
end
nParameters = numel(parameterNames);

%% Initialize or resume

if settings.resume && isfile(checkpointFile)
    try
        loaded = load(checkpointFile, 'rlResult');
        rlResult = loaded.rlResult;
        if ~isequal(rlResult.parameterNames, parameterNames)
            error('The checkpoint parameter definition does not match this model.')
        end
        rlResult = extendResult(rlResult, nSubjects, nParameters);
    catch checkpointError
        warning('ConflictHistoryRL:CheckpointRead', ...
            'Checkpoint could not be read (%s). Recovering subject fits.', ...
            checkpointError.message)
        rlResult = initializeResult(modelName, parameterNames, nSubjects);
        rlResult = recoverSubjectFits(rlResult, files, subjectOutputDir, modelName);
    end
else
    rlResult = initializeResult(modelName, parameterNames, nSubjects);
    if settings.resume
        rlResult = recoverSubjectFits(rlResult, files, subjectOutputDir, modelName);
    end
end
rlResult.settings = settings;

%% Participant fits

for subjectIndex = 1:nSubjects
    if rlResult.completed(subjectIndex)
        continue
    end

    subjectTimer = tic;
    subjectName = erase(files(subjectIndex).name, '.mat');
    fprintf('%s: subject %d/%d (%s)', ...
        modelName, subjectIndex, nSubjects, subjectName)

    loadedData = load(fullfile(dataDir, files(subjectIndex).name), 'all');
    assert(isfield(loadedData, 'all'), ...
        '%s does not contain an all structure.', files(subjectIndex).name)
    allData = loadedData.all;

    requiredFields = {'Congruence', 'Distance', 'RSI', 'Error'};
    if strcmp(outcomeType, 'RT')
        requiredFields = [requiredFields, {'LogRT', 'LinRT'}]; %#ok<AGROW>
    end
    assertFieldsPresent(allData, requiredFields, files(subjectIndex).name)
    nTrials = minimumFieldLength(allData, requiredFields);
    blockStart = identifyBlockStarts(allData, nTrials);

    fit = fitAlphaProfile(allData, nTrials, blockStart, ...
        outcomeType, settings);
    if ~isfinite(fit.negLogLike)
        warning('No finite fit for %s; leaving participant incomplete.', subjectName)
        continue
    end

    rlResult.subjectNames{subjectIndex} = subjectName;
    if strcmp(outcomeType, 'RT')
        rlResult.fitParams(:, subjectIndex) = ...
            [fit.alpha; fit.beta; log(fit.sigma)];
    else
        rlResult.fitParams(:, subjectIndex) = [fit.alpha; fit.beta];
    end
    rlResult.negLogLike(subjectIndex) = fit.negLogLike;
    rlResult.objective(subjectIndex) = fit.objective;
    rlResult.BIC(subjectIndex) = 2 * fit.negLogLike + ...
        nParameters * log(fit.nObservations);
    rlResult.nTrials(subjectIndex) = fit.nObservations;
    rlResult.completed(subjectIndex) = true;
    rlResult.alphaProfile{subjectIndex} = fit.alphaProfile;

    subjectFit = struct;
    subjectFit.subjectName = subjectName;
    subjectFit.modelName = modelName;
    subjectFit.parameterNames = parameterNames;
    subjectFit.fitParams = rlResult.fitParams(:, subjectIndex);
    subjectFit.negLogLike = fit.negLogLike;
    subjectFit.objective = fit.objective;
    subjectFit.BIC = rlResult.BIC(subjectIndex);
    subjectFit.validTrials = fit.validTrials;
    subjectFit.expectedConflict = fit.rl.expectedConflict;
    subjectFit.trialConflict = fit.rl.trialConflict;
    subjectFit.conflictPredictionError = fit.rl.predictionError;
    subjectFit.blockStart = blockStart;
    subjectFit.settings = settings;
    subjectFit.alphaProfile = fit.alphaProfile;
    if strcmp(outcomeType, 'RT')
        subjectFit.validRT = fit.validTrials;
        subjectFit.predictedLogRT = fit.predictedLogRT;
        subjectFit.residualLogRT = fit.residualLogRT;
    else
        subjectFit.validError = fit.validTrials;
        subjectFit.predictedError = fit.predictedError;
    end

    save(fullfile(subjectOutputDir, ...
        sprintf('%s_%s.mat', subjectName, modelName)), 'subjectFit')
    if mod(subjectIndex, settings.checkpointEvery) == 0 || ...
            subjectIndex == nSubjects
        saveCheckpoint(checkpointFile, rlResult)
    end
    fprintf(' (%.2f sec, alpha = %.3f, n = %d)\n', ...
        toc(subjectTimer), fit.alpha, fit.nObservations)
end

save(resultFile, 'rlResult', '-v7.3')
fprintf('Saved completed %s model to:\n%s\n', outcomeType, resultFile)
end

function result = recoverSubjectFits(result, files, subjectOutputDir, modelName)
% Rebuild the group checkpoint from independently saved participant files.
nRecovered = 0;
for subjectIndex = 1:numel(files)
    subjectName = erase(files(subjectIndex).name, '.mat');
    fitFile = fullfile(subjectOutputDir, ...
        sprintf('%s_%s.mat', subjectName, modelName));
    if ~isfile(fitFile), continue; end
    try
        loaded = load(fitFile, 'subjectFit');
        fit = loaded.subjectFit;
        if ~strcmp(fit.modelName, modelName) || ...
                numel(fit.fitParams) ~= size(result.fitParams, 1) || ...
                ~all(isfinite(fit.fitParams))
            continue
        end
        result.subjectNames{subjectIndex} = subjectName;
        result.fitParams(:, subjectIndex) = fit.fitParams(:);
        result.negLogLike(subjectIndex) = fit.negLogLike;
        result.BIC(subjectIndex) = fit.BIC;
        result.nTrials(subjectIndex) = sum(fit.validTrials);
        if isfield(fit, 'objective'), result.objective(subjectIndex) = fit.objective; end
        if isfield(fit, 'alphaProfile')
            result.alphaProfile{subjectIndex} = fit.alphaProfile;
        end
        result.completed(subjectIndex) = true;
        nRecovered = nRecovered + 1;
    catch
        % A single incomplete participant file should not block recovery.
    end
end
if nRecovered > 0
    fprintf('Recovered %d completed %s participant fits.\n', ...
        nRecovered, modelName)
end
end

function saveCheckpoint(checkpointFile, rlResult)
temporaryFile = [checkpointFile '.new.mat'];
save(temporaryFile, 'rlResult', '-v7.3')
[ok, message] = movefile(temporaryFile, checkpointFile, 'f');
assert(ok, 'Could not replace checkpoint: %s', message)
end

function settings = applyDefaults(settings, rootDir)

defaults.nSubjectsRequested = Inf;
defaults.minRT = 60;
defaults.maxRT = 1200;
defaults.resetAtBlock = true;
defaults.initialExpectedConflict = 0.5;
defaults.alphaCoarseGrid = unique([0.001, 0.05:0.05:0.95, 0.999]);
defaults.alphaTolerance = 1e-3;
% Retain the Beta(2,7) learning-rate prior used by the original
% RL_modeling_CongruencyOnly pipeline. It also prevents the nearly flat
% alpha/beta rescaling solution that can occur as alpha approaches zero.
defaults.useAlphaPrior = true;
defaults.alphaPriorA = 2;
defaults.alphaPriorB = 7;
defaults.ridgePenalty = 1e-4;
defaults.logisticMaxIterations = 100;
defaults.logisticTolerance = 1e-7;
defaults.resume = true;
defaults.checkpointEvery = 25;
defaults.modelStem = 'ConflictHistoryRL';
defaults.conflictDefinition = 'congruencyAndDistance';
defaults.dataDir = fullfile(rootDir, 'CC_Models', 'data');
defaults.outputRoot = fullfile(rootDir, 'CC_RL_Models', 'ConflictHistoryRL');

names = fieldnames(defaults);
for nameIndex = 1:numel(names)
    name = names{nameIndex};
    if ~isfield(settings, name)
        settings.(name) = defaults.(name);
    end
end
if ~isfolder(settings.dataDir)
    alternative = fullfile(rootDir, 'CC_models', 'data');
    if isfolder(alternative), settings.dataDir = alternative; end
end
end

function result = initializeResult(modelName, parameterNames, nSubjects)

nParameters = numel(parameterNames);
result = struct;
result.modelName = modelName;
result.parameterNames = parameterNames;
result.subjectNames = cell(nSubjects, 1);
result.fitParams = nan(nParameters, nSubjects);
result.negLogLike = nan(nSubjects, 1);
result.objective = nan(nSubjects, 1);
result.BIC = nan(nSubjects, 1);
result.nTrials = zeros(nSubjects, 1);
result.completed = false(nSubjects, 1);
result.alphaProfile = cell(nSubjects, 1);
end

function result = extendResult(result, nSubjects, nParameters)

oldN = size(result.fitParams, 2);
if oldN >= nSubjects, return; end
result.fitParams(:, oldN + 1:nSubjects) = nan(nParameters, nSubjects - oldN);
result.subjectNames(oldN + 1:nSubjects, 1) = {''};
result.negLogLike(oldN + 1:nSubjects, 1) = nan;
result.objective(oldN + 1:nSubjects, 1) = nan;
result.BIC(oldN + 1:nSubjects, 1) = nan;
result.nTrials(oldN + 1:nSubjects, 1) = 0;
result.completed(oldN + 1:nSubjects, 1) = false;
result.alphaProfile(oldN + 1:nSubjects, 1) = {[]};
end

function fit = fitAlphaProfile(allData, nTrials, blockStart, ...
    outcomeType, settings)

coarseAlpha = settings.alphaCoarseGrid(:);
coarseObjective = nan(size(coarseAlpha));
coarseNegLogLike = nan(size(coarseAlpha));
for candidateIndex = 1:numel(coarseAlpha)
    candidate = evaluateAlpha(coarseAlpha(candidateIndex), allData, ...
        nTrials, blockStart, outcomeType, settings);
    coarseObjective(candidateIndex) = candidate.objective;
    coarseNegLogLike(candidateIndex) = candidate.negLogLike;
end

[~, bestIndex] = min(coarseObjective);
lowerIndex = max(1, bestIndex - 1);
upperIndex = min(numel(coarseAlpha), bestIndex + 1);
lowerAlpha = coarseAlpha(lowerIndex);
upperAlpha = coarseAlpha(upperIndex);

if upperAlpha > lowerAlpha
    options = optimset('Display', 'off', 'TolX', settings.alphaTolerance);
    refinedAlpha = fminbnd(@(alpha)evaluateObjective(alpha, ...
        allData, nTrials, blockStart, outcomeType, settings), ...
        lowerAlpha, upperAlpha, options);
else
    refinedAlpha = coarseAlpha(bestIndex);
end

fit = evaluateAlpha(refinedAlpha, allData, nTrials, ...
    blockStart, outcomeType, settings);
fit.alphaProfile = table(coarseAlpha, coarseNegLogLike, ...
    coarseObjective, 'VariableNames', ...
    {'alpha', 'negLogLike', 'objective'});
end

function objective = evaluateObjective(alpha, allData, nTrials, ...
    blockStart, outcomeType, settings)

fit = evaluateAlpha(alpha, allData, nTrials, ...
    blockStart, outcomeType, settings);
objective = fit.objective;
end

function fit = evaluateAlpha(alpha, allData, nTrials, ...
    blockStart, outcomeType, settings)

rl = runConflictRL(allData, nTrials, blockStart, alpha, settings);
if strcmp(outcomeType, 'RT')
    [X, y, valid] = makeRTDesign(allData, rl, blockStart, settings);
    if numel(y) <= size(X, 2) || rank(X) < size(X, 2)
        fit = invalidFit(alpha, rl, nTrials, outcomeType);
        return
    end
    beta = X \ y;
    residual = y - X * beta;
    sigma = sqrt(max(mean(residual .^ 2), eps));
    n = numel(y);
    negLogLike = n * (0.5 * log(2 * pi) + log(sigma)) + ...
        sum(residual .^ 2) / (2 * sigma ^ 2);
    predicted = nan(nTrials, 1);
    predicted(valid) = X * beta;
    residualFull = nan(nTrials, 1);
    residualFull(valid) = residual;

    fit = struct('alpha', alpha, 'beta', beta, 'sigma', sigma, ...
        'negLogLike', negLogLike, 'objective', ...
        negLogLike + alphaPriorPenalty(alpha, settings), ...
        'nObservations', n, 'validTrials', valid, ...
        'predictedLogRT', predicted, 'residualLogRT', residualFull, ...
        'rl', rl);
else
    [X, y, valid] = makeErrorDesign(allData, rl, blockStart);
    if numel(y) <= size(X, 2) || numel(unique(y)) < 2 || ...
            rank(X) < size(X, 2)
        fit = invalidFit(alpha, rl, nTrials, outcomeType);
        return
    end
    [beta, negLogLike, penalizedNegLogLike] = ...
        fitLogisticModel(X, y, settings);
    predicted = nan(nTrials, 1);
    predicted(valid) = logistic(X * beta);
    fit = struct('alpha', alpha, 'beta', beta, ...
        'negLogLike', negLogLike, 'objective', ...
        penalizedNegLogLike + alphaPriorPenalty(alpha, settings), ...
        'nObservations', numel(y), 'validTrials', valid, ...
        'predictedError', predicted, 'rl', rl);
end
end

function rl = runConflictRL(allData, nTrials, blockStart, alpha, settings)

incongruent = columnDouble(allData.Congruence, nTrials);
incongruent = (incongruent + 1) / 2;
distance = columnDouble(allData.Distance, nTrials);
closeDistance = 1 - distance;
switch settings.conflictDefinition
    case 'congruencyAndDistance'
        trialConflict = (incongruent + closeDistance) / 2;
    case 'congruencyOnly'
        trialConflict = incongruent;
    otherwise
        error('Unknown conflict definition: %s', settings.conflictDefinition)
end

expectedConflict = nan(nTrials, 1);
predictionError = nan(nTrials, 1);
state = settings.initialExpectedConflict;
for trial = 1:nTrials
    if settings.resetAtBlock && blockStart(trial)
        state = settings.initialExpectedConflict;
    end
    expectedConflict(trial) = state;
    if isfinite(trialConflict(trial))
        predictionError(trial) = trialConflict(trial) - state;
        state = state + alpha * predictionError(trial);
        state = min(max(state, 0), 1);
    end
end

rl = struct('expectedConflict', expectedConflict, ...
    'trialConflict', trialConflict, ...
    'predictionError', predictionError);
end

function [X, y, valid] = makeRTDesign(allData, rl, blockStart, settings)

nTrials = numel(rl.expectedConflict);
logRT = columnDouble(allData.LogRT, nTrials);
linearRT = columnDouble(allData.LinRT, nTrials);
errorTrial = columnDouble(allData.Error, nTrials);
incongruent = (columnDouble(allData.Congruence, nTrials) + 1) / 2;
distance = columnDouble(allData.Distance, nTrials);
RSI = (columnDouble(allData.RSI, nTrials) + 1) / 2;
trialProgress = linspace(-0.5, 0.5, nTrials)';

previousIncongruence = [nan; incongruent(1:end - 1)];
previousError = [nan; errorTrial(1:end - 1)];
previousLogRT = [nan; logRT(1:end - 1)];
previousIncongruence(blockStart) = nan;
previousError(blockStart) = nan;
previousLogRT(blockStart) = nan;
previousLogRT = centerFinite(previousLogRT);

expectedCentered = centerFinite(rl.expectedConflict);
Xfull = [ones(nTrials, 1), expectedCentered, incongruent, ...
    expectedCentered .* incongruent, distance, RSI, trialProgress, ...
    previousIncongruence, previousError, previousLogRT];
valid = errorTrial == 0 & isfinite(logRT) & isfinite(linearRT) & ...
    linearRT > settings.minRT & linearRT < settings.maxRT & ...
    all(isfinite(Xfull), 2);
X = Xfull(valid, :);
y = logRT(valid);
end

function [X, y, valid] = makeErrorDesign(allData, rl, blockStart)

nTrials = numel(rl.expectedConflict);
errorTrial = columnDouble(allData.Error, nTrials);
incongruent = (columnDouble(allData.Congruence, nTrials) + 1) / 2;
distance = columnDouble(allData.Distance, nTrials);
RSI = (columnDouble(allData.RSI, nTrials) + 1) / 2;
trialProgress = linspace(-0.5, 0.5, nTrials)';

previousIncongruence = [nan; incongruent(1:end - 1)];
previousError = [nan; errorTrial(1:end - 1)];
previousIncongruence(blockStart) = nan;
previousError(blockStart) = nan;

expectedCentered = centerFinite(rl.expectedConflict);
Xfull = [ones(nTrials, 1), expectedCentered, incongruent, ...
    expectedCentered .* incongruent, distance, RSI, trialProgress, ...
    previousIncongruence, previousError];
valid = isfinite(errorTrial) & ismember(errorTrial, [0, 1]) & ...
    all(isfinite(Xfull), 2);
X = Xfull(valid, :);
y = errorTrial(valid);
end

function [beta, dataNegLogLike, penalizedNegLogLike] = ...
    fitLogisticModel(X, y, settings)

nParameters = size(X, 2);
beta = zeros(nParameters, 1);
penaltyMask = [0; ones(nParameters - 1, 1)];
for iteration = 1:settings.logisticMaxIterations
    eta = X * beta;
    probability = logistic(eta);
    weight = max(probability .* (1 - probability), 1e-8);
    gradient = X' * (probability - y) + ...
        settings.ridgePenalty .* penaltyMask .* beta;
    hessian = X' * (X .* weight) + ...
        settings.ridgePenalty .* diag(penaltyMask);
    step = hessian \ gradient;
    oldObjective = logisticObjective(beta, X, y, ...
        settings.ridgePenalty, penaltyMask);
    stepScale = 1;
    while stepScale > 1e-6
        candidateBeta = beta - stepScale * step;
        candidateObjective = logisticObjective(candidateBeta, X, y, ...
            settings.ridgePenalty, penaltyMask);
        if candidateObjective <= oldObjective, break; end
        stepScale = stepScale / 2;
    end
    betaChange = max(abs(stepScale * step));
    beta = beta - stepScale * step;
    if betaChange < settings.logisticTolerance, break; end
end

eta = X * beta;
dataNegLogLike = sum(softplus(eta) - y .* eta);
penalizedNegLogLike = dataNegLogLike + ...
    0.5 * settings.ridgePenalty * sum((penaltyMask .* beta) .^ 2);
end

function value = logisticObjective(beta, X, y, ridgePenalty, penaltyMask)

eta = X * beta;
value = sum(softplus(eta) - y .* eta) + ...
    0.5 * ridgePenalty * sum((penaltyMask .* beta) .^ 2);
end

function values = softplus(values)
values = max(values, 0) + log1p(exp(-abs(values)));
end

function probability = logistic(values)
probability = zeros(size(values));
positive = values >= 0;
probability(positive) = 1 ./ (1 + exp(-values(positive)));
expValues = exp(values(~positive));
probability(~positive) = expValues ./ (1 + expValues);
end

function penalty = alphaPriorPenalty(alpha, settings)
if ~settings.useAlphaPrior
    penalty = 0;
    return
end
logDensity = (settings.alphaPriorA - 1) * log(alpha) + ...
    (settings.alphaPriorB - 1) * log(1 - alpha);
penalty = -logDensity;
end

function fit = invalidFit(alpha, rl, nTrials, outcomeType)
if strcmp(outcomeType, 'RT')
    fit = struct('alpha', alpha, 'beta', nan(10, 1), 'sigma', nan, ...
        'negLogLike', inf, 'objective', inf, 'nObservations', 0, ...
        'validTrials', false(nTrials, 1), ...
        'predictedLogRT', nan(nTrials, 1), ...
        'residualLogRT', nan(nTrials, 1), 'rl', rl);
else
    fit = struct('alpha', alpha, 'beta', nan(9, 1), ...
        'negLogLike', inf, 'objective', inf, 'nObservations', 0, ...
        'validTrials', false(nTrials, 1), ...
        'predictedError', nan(nTrials, 1), 'rl', rl);
end
end

function blockStart = identifyBlockStarts(allData, nTrials)
blockStart = false(nTrials, 1);
blockStart(1) = true;
if ~isfield(allData, 'BlockTrnr'), return; end
blockTrial = columnDouble(allData.BlockTrnr, nTrials);
blockStart(2:end) = ~isfinite(blockTrial(2:end)) | ...
    ~isfinite(blockTrial(1:end - 1)) | ...
    blockTrial(2:end) <= blockTrial(1:end - 1);
end

function centered = centerFinite(values)
centered = values;
finiteValues = isfinite(values);
if any(finiteValues)
    centered(finiteValues) = values(finiteValues) - mean(values(finiteValues));
end
end

function values = columnDouble(values, nValues)
values = double(values(1:nValues));
values = values(:);
end

function nTrials = minimumFieldLength(allData, fieldNames)
fieldLengths = zeros(numel(fieldNames), 1);
for fieldIndex = 1:numel(fieldNames)
    fieldLengths(fieldIndex) = numel(allData.(fieldNames{fieldIndex}));
end
nTrials = min(fieldLengths);
end

function assertFieldsPresent(allData, fieldNames, fileName)
missingFields = fieldNames(~isfield(allData, fieldNames));
assert(isempty(missingFields), '%s is missing fields: %s', ...
    fileName, strjoin(missingFields, ', '))
end
