function dbmResult = fitTargetSequenceDirectionalDBM( ...
    outcomeType, rootDir, settings)
%FITTARGETSEQUENCEDIRECTIONALDBM Fit RT or error directional target DBM.
%
% The model has two dynamic beliefs:
%   thetaLeft  = P(current target is right | previous target was left)
%   thetaRight = P(current target is right | previous target was right)
%
% A shared alpha controls persistence of both beliefs. On every valid
% transition both beliefs decay toward Beta(1,1), and only the belief
% selected by the previous target receives the current likelihood update.

if nargin < 1 || isempty(outcomeType)
    outcomeType = 'RT';
end
if nargin < 2 || isempty(rootDir)
    rootDir = fileparts(mfilename('fullpath'));
end
if nargin < 3
    settings = struct;
end

outcomeType = validatestring(outcomeType, {'RT', 'Error'});
settings = applyDefaultSettings(settings, outcomeType);

dataDir = fullfile(rootDir, 'CC_Models', 'data');
if ~isfolder(dataDir)
    dataDir = fullfile(rootDir, 'CC_models', 'data');
end
assert(isfolder(dataDir), 'Behavioral data directory was not found.')

outputDir = fullfile(rootDir, 'CC_RL_Models', ...
    'TargetSequenceDirectionalDBM', outcomeType);
subjectOutputDir = fullfile(outputDir, 'SubjectData');
if ~isfolder(subjectOutputDir)
    mkdir(subjectOutputDir)
end

modelName = sprintf('TargetSequenceDirectionalDBM_%s', outcomeType);
resultFile = fullfile(outputDir, sprintf('%s_results.mat', modelName));
checkpointFile = fullfile(outputDir, ...
    sprintf('%s_results_incomplete.mat', modelName));

files = dir(fullfile(dataDir, '*.mat'));
assert(~isempty(files), 'No subject .mat files were found in %s.', dataDir)
[~, fileOrder] = sort({files.name});
files = files(fileOrder);
files = files(1:min(settings.nSubjectsRequested, numel(files)));
nSubjects = numel(files);

if strcmp(outcomeType, 'RT')
    parameterNames = {'alpha', 'intc', 'bUnexpected', ...
        'bAlternation', 'bCongruence', ...
        'bUnexpectedXCongruence', 'bDistance', 'bRSI', 'bTrial', ...
        'bTargetRight', 'bPreviousLogRT', 'logSigma'};
else
    parameterNames = {'alpha', 'intc', 'bUnexpected', ...
        'bAlternation', 'bCongruence', ...
        'bUnexpectedXCongruence', 'bDistance', 'bRSI', 'bTrial', ...
        'bTargetRight', 'bPreviousError'};
end
nParameters = numel(parameterNames);

dbmResult = initializeOrResume(checkpointFile, settings, modelName, ...
    parameterNames, nParameters, nSubjects);

for subjectIndex = 1:nSubjects

    if dbmResult.completed(subjectIndex)
        continue
    end

    subjectTimer = tic;
    subjectName = erase(files(subjectIndex).name, '.mat');
    fprintf('%s directional DBM: subject %d/%d (%s)', ...
        outcomeType, subjectIndex, nSubjects, subjectName)

    loadedData = load(fullfile(dataDir, files(subjectIndex).name), 'all');
    assert(isfield(loadedData, 'all'), ...
        '%s does not contain an all structure.', files(subjectIndex).name)
    allData = loadedData.all;

    if strcmp(outcomeType, 'RT')
        requiredFields = {'Target', 'LogRT', 'LinRT', 'Error', ...
            'Congruence', 'Distance', 'RSI'};
    else
        requiredFields = {'Target', 'Error', 'Congruence', ...
            'Distance', 'RSI'};
    end
    assertFieldsPresent(allData, requiredFields, files(subjectIndex).name)

    nTrialsAll = minimumFieldLength(allData, requiredFields);
    target = double(allData.Target(1:nTrialsAll));
    target = target(:);
    blockStart = identifyBlockStarts(allData, nTrialsAll);

    fit = fitAlphaProfile(outcomeType, allData, target, ...
        blockStart, settings);

    dbmResult.subjectNames{subjectIndex} = subjectName;
    if strcmp(outcomeType, 'RT')
        dbmResult.fitParams(:, subjectIndex) = ...
            [fit.alpha; fit.beta; log(fit.sigma)];
    else
        dbmResult.fitParams(:, subjectIndex) = [fit.alpha; fit.beta];
    end
    dbmResult.negLogLike(subjectIndex) = fit.negLogLike;
    dbmResult.objective(subjectIndex) = fit.objective;
    dbmResult.BIC(subjectIndex) = 2 * fit.negLogLike + ...
        nParameters * log(fit.nObservations);
    dbmResult.nTrials(subjectIndex) = fit.nObservations;
    dbmResult.alphaProfile{subjectIndex} = fit.alphaProfile;
    dbmResult.completed(subjectIndex) = true;

    subjectFit = makeSubjectFit(subjectName, modelName, parameterNames, ...
        dbmResult.fitParams(:, subjectIndex), dbmResult.BIC(subjectIndex), ...
        fit, blockStart, settings, outcomeType);
    subjectFile = fullfile(subjectOutputDir, sprintf('%s_%s.mat', ...
        subjectName, modelName));
    save(subjectFile, 'subjectFit')

    save(checkpointFile, 'dbmResult', '-v7.3')
    fprintf(' (%.2f sec, alpha = %.3f, n = %d)\n', ...
        toc(subjectTimer), fit.alpha, fit.nObservations)
end

save(resultFile, 'dbmResult', '-v7.3')
fprintf('Saved completed %s directional model to:\n%s\n', ...
    lower(outcomeType), resultFile)
end

function settings = applyDefaultSettings(settings, outcomeType)

defaults.nSubjectsRequested = Inf;
defaults.minRT = 60;
defaults.maxRT = 1200;
defaults.thetaGridSize = 201;
defaults.betaPriorA = 1;
defaults.betaPriorB = 1;
defaults.resetAtBlock = true;
defaults.alphaCoarseGrid = unique([0.001, 0.05:0.05:0.95, 0.999]);
defaults.alphaTolerance = 1e-3;
defaults.useAlphaPrior = false;
defaults.alphaPriorA = 2;
defaults.alphaPriorB = 2;
defaults.ridgePenalty = 1e-4;
defaults.logisticMaxIterations = 100;
defaults.logisticTolerance = 1e-7;
defaults.resume = true;

fieldNames = fieldnames(defaults);
for fieldIndex = 1:numel(fieldNames)
    fieldName = fieldNames{fieldIndex};
    if ~isfield(settings, fieldName) || isempty(settings.(fieldName))
        settings.(fieldName) = defaults.(fieldName);
    end
end
settings.outcomeType = outcomeType;
settings.transitionModel = 'direction-specific Markov order 1';
settings.sharedAlpha = true;
settings.inactiveContextDecay = true;
end

function result = initializeOrResume(checkpointFile, settings, ...
    modelName, parameterNames, nParameters, nSubjects)

if settings.resume && isfile(checkpointFile)
    loadedCheckpoint = load(checkpointFile, 'dbmResult');
    result = loadedCheckpoint.dbmResult;
    assert(strcmp(result.modelName, modelName), ...
        'Checkpoint model does not match %s.', modelName)

    if size(result.fitParams, 2) < nSubjects
        result.fitParams(:, end + 1:nSubjects) = nan;
        result.negLogLike(end + 1:nSubjects, 1) = nan;
        result.objective(end + 1:nSubjects, 1) = nan;
        result.BIC(end + 1:nSubjects, 1) = nan;
        result.nTrials(end + 1:nSubjects, 1) = 0;
        result.completed(end + 1:nSubjects, 1) = false;
        result.alphaProfile(end + 1:nSubjects, 1) = {[]};
        result.subjectNames(end + 1:nSubjects, 1) = {''};
    end
else
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
result.settings = settings;
end

function subjectFit = makeSubjectFit(subjectName, modelName, ...
    parameterNames, fitParams, BIC, fit, blockStart, settings, outcomeType)

subjectFit = struct;
subjectFit.subjectName = subjectName;
subjectFit.modelName = modelName;
subjectFit.parameterNames = parameterNames;
subjectFit.fitParams = fitParams;
subjectFit.negLogLike = fit.negLogLike;
subjectFit.BIC = BIC;

if strcmp(outcomeType, 'RT')
    subjectFit.validRT = fit.validTrials;
    subjectFit.predictedLogRT = fit.predictedLogRT;
    subjectFit.residualLogRT = fit.residualLogRT;
else
    subjectFit.validError = fit.validTrials;
    subjectFit.predictedError = fit.predictedError;
end

dbmFields = fieldnames(fit.dbm);
for fieldIndex = 1:numel(dbmFields)
    fieldName = dbmFields{fieldIndex};
    subjectFit.(fieldName) = fit.dbm.(fieldName);
end
subjectFit.blockStart = blockStart;
subjectFit.settings = settings;
end

function fit = fitAlphaProfile(outcomeType, allData, target, ...
    blockStart, settings)

coarseAlpha = settings.alphaCoarseGrid(:);
coarseObjective = nan(size(coarseAlpha));
coarseNegLogLike = nan(size(coarseAlpha));

for candidateIndex = 1:numel(coarseAlpha)
    candidateFit = evaluateAlpha(coarseAlpha(candidateIndex), ...
        outcomeType, allData, target, blockStart, settings);
    coarseObjective(candidateIndex) = candidateFit.objective;
    coarseNegLogLike(candidateIndex) = candidateFit.negLogLike;
end

[~, bestCoarseIndex] = min(coarseObjective);
lowerIndex = max(1, bestCoarseIndex - 1);
upperIndex = min(numel(coarseAlpha), bestCoarseIndex + 1);
lowerAlpha = coarseAlpha(lowerIndex);
upperAlpha = coarseAlpha(upperIndex);

if upperAlpha > lowerAlpha
    optimizerOptions = optimset('Display', 'off', ...
        'TolX', settings.alphaTolerance);
    refinedAlpha = fminbnd(@(alpha)evaluateObjective(alpha, ...
        outcomeType, allData, target, blockStart, settings), ...
        lowerAlpha, upperAlpha, optimizerOptions);
else
    refinedAlpha = coarseAlpha(bestCoarseIndex);
end

fit = evaluateAlpha(refinedAlpha, outcomeType, allData, target, ...
    blockStart, settings);
fit.alphaProfile = table(coarseAlpha, coarseNegLogLike, ...
    coarseObjective, 'VariableNames', ...
    {'alpha', 'negLogLike', 'objective'});
end

function objective = evaluateObjective(alpha, outcomeType, allData, ...
    target, blockStart, settings)

candidateFit = evaluateAlpha(alpha, outcomeType, allData, target, ...
    blockStart, settings);
objective = candidateFit.objective;
end

function fit = evaluateAlpha(alpha, outcomeType, allData, target, ...
    blockStart, settings)

dbm = runDirectionalTargetDBM(target, blockStart, alpha, settings);

if strcmp(outcomeType, 'RT')
    [X, y, validTrials] = makeRTDesign( ...
        allData, dbm, blockStart, settings);
    if invalidDesign(X, y, false)
        fit = invalidFit(alpha, dbm, numel(target), outcomeType);
        return
    end

    beta = X \ y;
    residual = y - X * beta;
    sigma = sqrt(max(mean(residual .^ 2), eps));
    nObservations = numel(y);
    negLogLike = nObservations * ...
        (0.5 * log(2 * pi) + log(sigma)) + ...
        sum(residual .^ 2) / (2 * sigma ^ 2);
    objective = negLogLike + alphaPriorPenalty(alpha, settings);

    predictedLogRT = nan(numel(target), 1);
    predictedLogRT(validTrials) = X * beta;
    residualLogRT = nan(numel(target), 1);
    residualLogRT(validTrials) = residual;

    fit = struct('alpha', alpha, 'beta', beta, 'sigma', sigma, ...
        'negLogLike', negLogLike, 'objective', objective, ...
        'nObservations', nObservations, 'validTrials', validTrials, ...
        'predictedLogRT', predictedLogRT, ...
        'residualLogRT', residualLogRT, 'dbm', dbm);
else
    [X, y, validTrials] = makeErrorDesign(allData, dbm, blockStart);
    if invalidDesign(X, y, true)
        fit = invalidFit(alpha, dbm, numel(target), outcomeType);
        return
    end

    [beta, negLogLike, penalizedNegLogLike] = ...
        fitLogisticModel(X, y, settings);
    objective = penalizedNegLogLike + ...
        alphaPriorPenalty(alpha, settings);
    predictedError = nan(numel(target), 1);
    predictedError(validTrials) = logistic(X * beta);

    fit = struct('alpha', alpha, 'beta', beta, ...
        'negLogLike', negLogLike, 'objective', objective, ...
        'nObservations', numel(y), 'validTrials', validTrials, ...
        'predictedError', predictedError, 'dbm', dbm);
end
end

function invalid = invalidDesign(X, y, requireBinaryVariation)

invalid = numel(y) <= size(X, 2) || rank(X) < size(X, 2);
if requireBinaryVariation
    invalid = invalid || numel(unique(y)) < 2;
end
end

function [X, y, valid] = makeRTDesign(allData, dbm, ...
    blockStart, settings)

nTrials = numel(dbm.pObserved);
logRT = columnDouble(allData.LogRT, nTrials);
linearRT = columnDouble(allData.LinRT, nTrials);
errorTrial = columnDouble(allData.Error, nTrials);
incongruent = (columnDouble(allData.Congruence, nTrials) + 1) / 2;
distance = columnDouble(allData.Distance, nTrials);
RSI = (columnDouble(allData.RSI, nTrials) + 1) / 2;
targetRight = (columnDouble(allData.Target, nTrials) + 1) / 2;
trialProgress = linspace(-0.5, 0.5, nTrials)';

previousLogRT = [nan; logRT(1:end - 1)];
previousLogRT(blockStart) = nan;
previousLogRT = centerFinite(previousLogRT);

unexpectednessCentered = centerFinite(dbm.unexpectedness);
alternation = 1 - dbm.isRepeat;

Xfull = [ones(nTrials, 1), unexpectednessCentered, alternation, ...
    incongruent, unexpectednessCentered .* incongruent, distance, RSI, ...
    trialProgress, targetRight, previousLogRT];

valid = errorTrial == 0 & isfinite(logRT) & isfinite(linearRT) & ...
    linearRT > settings.minRT & linearRT < settings.maxRT & ...
    all(isfinite(Xfull), 2);
X = Xfull(valid, :);
y = logRT(valid);
end

function [X, y, valid] = makeErrorDesign(allData, dbm, blockStart)

nTrials = numel(dbm.pObserved);
errorTrial = columnDouble(allData.Error, nTrials);
incongruent = (columnDouble(allData.Congruence, nTrials) + 1) / 2;
distance = columnDouble(allData.Distance, nTrials);
RSI = (columnDouble(allData.RSI, nTrials) + 1) / 2;
targetRight = (columnDouble(allData.Target, nTrials) + 1) / 2;
trialProgress = linspace(-0.5, 0.5, nTrials)';

previousError = [nan; errorTrial(1:end - 1)];
previousError(blockStart) = nan;

unexpectednessCentered = centerFinite(dbm.unexpectedness);
alternation = 1 - dbm.isRepeat;

Xfull = [ones(nTrials, 1), unexpectednessCentered, alternation, ...
    incongruent, unexpectednessCentered .* incongruent, distance, RSI, ...
    trialProgress, targetRight, previousError];

valid = isfinite(errorTrial) & ismember(errorTrial, [0, 1]) & ...
    all(isfinite(Xfull), 2);
X = Xfull(valid, :);
y = errorTrial(valid);
end

function dbm = runDirectionalTargetDBM(target, blockStart, ...
    alpha, settings)

nTrials = numel(target);
theta = linspace(1e-4, 1 - 1e-4, settings.thetaGridSize)';
logPrior = (settings.betaPriorA - 1) .* log(theta) + ...
    (settings.betaPriorB - 1) .* log(1 - theta);
p0 = exp(logPrior - max(logPrior));
p0 = p0 / sum(p0);

posteriorAfterLeft = p0;
posteriorAfterRight = p0;

pRight = nan(nTrials, 1);
pRightAfterLeft = nan(nTrials, 1);
pRightAfterRight = nan(nTrials, 1);
pRepeat = nan(nTrials, 1);
pObserved = nan(nTrials, 1);
isRepeat = nan(nTrials, 1);
directionalPredictionError = nan(nTrials, 1);
repetitionPredictionError = nan(nTrials, 1);

for trial = 2:nTrials

    if (settings.resetAtBlock && blockStart(trial)) || ...
            ~isfinite(target(trial - 1)) || ~isfinite(target(trial))
        posteriorAfterLeft = p0;
        posteriorAfterRight = p0;
        continue
    end

    predictiveAfterLeft = alpha .* posteriorAfterLeft + ...
        (1 - alpha) .* p0;
    predictiveAfterLeft = predictiveAfterLeft / ...
        sum(predictiveAfterLeft);
    predictiveAfterRight = alpha .* posteriorAfterRight + ...
        (1 - alpha) .* p0;
    predictiveAfterRight = predictiveAfterRight / ...
        sum(predictiveAfterRight);

    pRightAfterLeft(trial) = sum(theta .* predictiveAfterLeft);
    pRightAfterRight(trial) = sum(theta .* predictiveAfterRight);

    previousRight = target(trial - 1) > 0;
    currentRight = double(target(trial) > 0);
    if previousRight
        predictiveActive = predictiveAfterRight;
        pRight(trial) = pRightAfterRight(trial);
        pRepeat(trial) = pRight(trial);
    else
        predictiveActive = predictiveAfterLeft;
        pRight(trial) = pRightAfterLeft(trial);
        pRepeat(trial) = 1 - pRight(trial);
    end

    isRepeat(trial) = double(target(trial) == target(trial - 1));
    if currentRight
        observationLikelihood = theta;
        pObserved(trial) = pRight(trial);
    else
        observationLikelihood = 1 - theta;
        pObserved(trial) = 1 - pRight(trial);
    end

    directionalPredictionError(trial) = ...
        currentRight - pRight(trial);
    repetitionPredictionError(trial) = ...
        isRepeat(trial) - pRepeat(trial);

    updatedActive = observationLikelihood .* predictiveActive;
    updatedActive = updatedActive / sum(updatedActive);

    if previousRight
        posteriorAfterRight = updatedActive;
        posteriorAfterLeft = predictiveAfterLeft;
    else
        posteriorAfterLeft = updatedActive;
        posteriorAfterRight = predictiveAfterRight;
    end
end

dbm = struct;
dbm.pRight = pRight;
dbm.pRightAfterLeft = pRightAfterLeft;
dbm.pRightAfterRight = pRightAfterRight;
dbm.directionalExpectation = 2 .* pRight - 1;
dbm.pRepeat = pRepeat;
dbm.pObserved = pObserved;
dbm.unexpectedness = 1 - pObserved;
dbm.surprise = -log(max(pObserved, realmin));
dbm.predictionError = directionalPredictionError;
dbm.repetitionPredictionError = repetitionPredictionError;
dbm.isRepeat = isRepeat;
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
    oldObjective = logisticObjective( ...
        beta, X, y, settings.ridgePenalty, penaltyMask);
    stepScale = 1;

    while stepScale > 1e-6
        candidateBeta = beta - stepScale .* step;
        candidateObjective = logisticObjective(candidateBeta, X, y, ...
            settings.ridgePenalty, penaltyMask);
        if candidateObjective <= oldObjective
            break
        end
        stepScale = stepScale / 2;
    end

    betaChange = max(abs(stepScale .* step));
    beta = beta - stepScale .* step;
    if betaChange < settings.logisticTolerance
        break
    end
end

eta = X * beta;
dataNegLogLike = sum(softplus(eta) - y .* eta);
penalizedNegLogLike = dataNegLogLike + ...
    0.5 * settings.ridgePenalty * ...
    sum((penaltyMask .* beta) .^ 2);
end

function objective = logisticObjective(beta, X, y, ...
    ridgePenalty, penaltyMask)

eta = X * beta;
objective = sum(softplus(eta) - y .* eta) + ...
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

function fit = invalidFit(alpha, dbm, nTrials, outcomeType)

fit = struct('alpha', alpha, 'beta', nan(10, 1), ...
    'negLogLike', inf, 'objective', inf, 'nObservations', 0, ...
    'validTrials', false(nTrials, 1), 'dbm', dbm);
if strcmp(outcomeType, 'RT')
    fit.sigma = nan;
    fit.predictedLogRT = nan(nTrials, 1);
    fit.residualLogRT = nan(nTrials, 1);
else
    fit.predictedError = nan(nTrials, 1);
end
end

function blockStart = identifyBlockStarts(allData, nTrials)

blockStart = false(nTrials, 1);
blockStart(1) = true;
if ~isfield(allData, 'BlockTrnr')
    return
end
blockTrial = columnDouble(allData.BlockTrnr, nTrials);
blockStart(2:end) = ~isfinite(blockTrial(2:end)) | ...
    ~isfinite(blockTrial(1:end - 1)) | ...
    blockTrial(2:end) <= blockTrial(1:end - 1);
end

function centered = centerFinite(values)

centered = values;
finiteValues = isfinite(values);
if any(finiteValues)
    centered(finiteValues) = values(finiteValues) - ...
        mean(values(finiteValues));
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
