%% Parametric predictive checks for the target-sequence DBM RT model
%
% This script:
%   1) Loads completed subject-level fits from DBM_TargetSequence_RT.m
%   2) Simulates replicated RT data from each fitted log-normal RT model
%   3) Applies the same t0:t-3 target-sequence bins used by
%      PlotSequenceEffects_RT_pHand.m
%   4) Compares observed sequence effects with predictive distributions
%   5) Plots residual sequence effects and pattern-confirmation contrasts
%
% The predictive intervals condition on the fitted parameter estimates.
% They contain trial-level observation noise but not uncertainty in alpha
% or the regression coefficients. They are therefore parametric
% predictive checks rather than fully Bayesian posterior predictive checks.

if ~exist('ppcConfig', 'var') || ~isstruct(ppcConfig)
    ppcConfig = struct;
end
clearvars -except ppcConfig
clc

rootDir = fileparts(mfilename('fullpath'));
if isempty(rootDir)
    rootDir = pwd;
end

%% Settings

settings.nReplications = 500;
settings.randomSeed = 20260730;
settings.historyLength = 3;
settings.nSubjectsRequested = Inf;
settings.excludeCrossBlockHistories = true;
settings.resampleOutsideRTRange = true;
settings.saveFigures = true;
settings.figureVisible = 'on';

if isfield(ppcConfig, 'settings')
    overrideFields = fieldnames(ppcConfig.settings);
    for overrideIndex = 1:numel(overrideFields)
        overrideName = overrideFields{overrideIndex};
        settings.(overrideName) = ppcConfig.settings.(overrideName);
    end
end

dataDir = fullfile(rootDir, 'CC_Models', 'data');
if ~isfolder(dataDir)
    dataDir = fullfile(rootDir, 'CC_models', 'data');
end

fitDir = fullfile(rootDir, 'CC_RL_Models', ...
    'TargetSequenceDBM', 'RT', 'SubjectData');
outputDir = fullfile(rootDir, 'Figures', 'TargetSequenceDBM', 'RT_PPC');
fitPattern = '*_TargetSequenceDBM_RT.mat';
outputStem = 'PPC_TargetSequenceDBM_RT';

if isfield(ppcConfig, 'fitDir'), fitDir = ppcConfig.fitDir; end
if isfield(ppcConfig, 'outputDir'), outputDir = ppcConfig.outputDir; end
if isfield(ppcConfig, 'fitPattern'), fitPattern = ppcConfig.fitPattern; end
if isfield(ppcConfig, 'outputStem'), outputStem = ppcConfig.outputStem; end

assert(isfolder(dataDir), 'Behavioral data directory was not found.')
assert(isfolder(fitDir), 'RT DBM subject-fit directory was not found.')
if ~isfolder(outputDir)
    mkdir(outputDir)
end

rng(settings.randomSeed, 'twister')

%% Find completed fits and prepare sequence definitions

fitFiles = dir(fullfile(fitDir, fitPattern));
assert(~isempty(fitFiles), 'No completed subject-level RT fits were found.')
[~, fitOrder] = sort({fitFiles.name});
fitFiles = fitFiles(fitOrder);
fitFiles = fitFiles(1:min(settings.nSubjectsRequested, numel(fitFiles)));

nSubjectsAvailable = numel(fitFiles);
[sequenceLabels, sequencePowers] = ...
    makeSequenceLabels(settings.historyLength + 1);
nSequences = numel(sequenceLabels);

conditionLabels = {'Congruent', 'Incongruent'};
conditionColors = [0.18 0.45 0.72; 0.82 0.33 0.27];
nConditions = numel(conditionLabels);

% The preceding two transitions establish a repetition or alternation
% pattern; the current transition either confirms or violates it.
patternLabels = {'RR -> R', 'RR -> A', 'AA -> A', 'AA -> R'};
nPatterns = numel(patternLabels);

observedSubjectMeans = ...
    nan(nSubjectsAvailable, nSequences, nConditions);
residualSubjectMeans = ...
    nan(nSubjectsAvailable, nSequences, nConditions);
observedPatternSubjectMeans = ...
    nan(nSubjectsAvailable, nPatterns, nConditions);

replicatedSequenceSum = ...
    zeros(settings.nReplications, nSequences, nConditions);
replicatedSequenceSubjectCount = zeros(nSequences, nConditions);
replicatedPatternSum = ...
    zeros(settings.nReplications, nPatterns, nConditions);
replicatedPatternSubjectCount = zeros(nPatterns, nConditions);

subjectNames = strings(nSubjectsAvailable, 1);
subjectsIncluded = false(nSubjectsAvailable, 1);

%% Simulate participants and summarize target-sequence cells

for subjectIndex = 1:nSubjectsAvailable

    if subjectIndex == 1 || mod(subjectIndex, 25) == 0 || ...
            subjectIndex == nSubjectsAvailable
        fprintf('RT predictive check: subject %d/%d\n', ...
            subjectIndex, nSubjectsAvailable)
    end

    loadedFit = load(fullfile(fitDir, fitFiles(subjectIndex).name), ...
        'subjectFit');
    if ~isfield(loadedFit, 'subjectFit')
        warning('%s does not contain subjectFit; skipping.', ...
            fitFiles(subjectIndex).name)
        continue
    end
    subjectFit = loadedFit.subjectFit;
    subjectName = string(subjectFit.subjectName);
    subjectNames(subjectIndex) = subjectName;

    behavioralFile = fullfile(dataDir, char(subjectName + ".mat"));
    if ~isfile(behavioralFile)
        warning('Behavioral data for %s were not found; skipping.', ...
            subjectName)
        continue
    end

    loadedData = load(behavioralFile, 'all');
    if ~isfield(loadedData, 'all')
        warning('%s does not contain all; skipping.', behavioralFile)
        continue
    end
    allData = loadedData.all;

    requiredFields = {'Target', 'LinRT', 'Congruence'};
    assertFieldsPresent(allData, requiredFields, behavioralFile)

    sigmaIndex = find(strcmp(subjectFit.parameterNames, 'logSigma'), 1);
    if isempty(sigmaIndex)
        warning('logSigma was not found for %s; skipping.', subjectName)
        continue
    end
    sigma = exp(subjectFit.fitParams(sigmaIndex));

    nTrials = min([numel(allData.Target), numel(allData.LinRT), ...
        numel(allData.Congruence), ...
        numel(subjectFit.predictedLogRT), ...
        numel(subjectFit.residualLogRT), ...
        numel(subjectFit.validRT), ...
        numel(subjectFit.blockStart)]);

    if nTrials <= settings.historyLength || ~isfinite(sigma)
        warning('Insufficient or invalid fitted data for %s; skipping.', ...
            subjectName)
        continue
    end

    target = double(allData.Target(1:nTrials));
    target = target(:);
    observedRT = double(allData.LinRT(1:nTrials));
    observedRT = observedRT(:);
    congruence = double(allData.Congruence(1:nTrials));
    congruence = congruence(:);
    predictedLogRT = double(subjectFit.predictedLogRT(1:nTrials));
    predictedLogRT = predictedLogRT(:);
    residualLogRT = double(subjectFit.residualLogRT(1:nTrials));
    residualLogRT = residualLogRT(:);
    modelValid = logical(subjectFit.validRT(1:nTrials));
    modelValid = modelValid(:);
    blockStart = logical(subjectFit.blockStart(1:nTrials));
    blockStart = blockStart(:);

    [sequenceCode, currentTrial] = encodeSequenceThroughCurrent( ...
        target, settings.historyLength, sequencePowers);

    currentTarget = target(currentTrial);
    currentRT = observedRT(currentTrial);
    currentCongruence = congruence(currentTrial);
    currentMu = predictedLogRT(currentTrial);
    currentResidual = residualLogRT(currentTrial);
    currentModelValid = modelValid(currentTrial);

    historyWithinBlock = true(size(currentTrial));
    if settings.excludeCrossBlockHistories
        blockID = cumsum(blockStart);
        for lag = 1:settings.historyLength
            historyWithinBlock = historyWithinBlock & ...
                blockID(currentTrial) == blockID(currentTrial - lag);
        end
    end

    baseValid = currentModelValid & historyWithinBlock & ...
        isfinite(sequenceCode) & isfinite(currentTarget) & ...
        isfinite(currentRT) & isfinite(currentCongruence) & ...
        isfinite(currentMu) & isfinite(currentResidual);

    if ~any(baseValid)
        warning('No valid predictive-check trials for %s; skipping.', ...
            subjectName)
        continue
    end

    [minimumRT, maximumRT] = subjectRTRange(subjectFit);
    simulatedRT = nan(numel(currentTrial), settings.nReplications);
    simulatedRT(baseValid, :) = simulateLogNormalRT( ...
        currentMu(baseValid), sigma, settings.nReplications, ...
        minimumRT, maximumRT, settings.resampleOutsideRTRange);

    patternCode = encodePatternConfirmation(target, currentTrial);

    for condition = 1:nConditions
        congruenceCode = 2 * condition - 3;

        for sequence = 1:nSequences
            useTrials = baseValid & ...
                sequenceCode == (sequence - 1) & ...
                currentCongruence == congruenceCode;

            if any(useTrials)
                observedSubjectMeans(subjectIndex, sequence, condition) = ...
                    mean(currentRT(useTrials));
                residualSubjectMeans(subjectIndex, sequence, condition) = ...
                    mean(currentResidual(useTrials));

                simulatedSubjectMean = ...
                    mean(simulatedRT(useTrials, :), 1)';
                replicatedSequenceSum(:, sequence, condition) = ...
                    replicatedSequenceSum(:, sequence, condition) + ...
                    simulatedSubjectMean;
                replicatedSequenceSubjectCount(sequence, condition) = ...
                    replicatedSequenceSubjectCount(sequence, condition) + 1;
            end
        end

        for pattern = 1:nPatterns
            useTrials = baseValid & patternCode == pattern & ...
                currentCongruence == congruenceCode;

            if any(useTrials)
                observedPatternSubjectMeans( ...
                    subjectIndex, pattern, condition) = ...
                    mean(currentRT(useTrials));

                simulatedSubjectMean = ...
                    mean(simulatedRT(useTrials, :), 1)';
                replicatedPatternSum(:, pattern, condition) = ...
                    replicatedPatternSum(:, pattern, condition) + ...
                    simulatedSubjectMean;
                replicatedPatternSubjectCount(pattern, condition) = ...
                    replicatedPatternSubjectCount(pattern, condition) + 1;
            end
        end
    end

    subjectsIncluded(subjectIndex) = true;
end

%% Group summaries and predictive intervals

subjectNames = subjectNames(subjectsIncluded);
observedSubjectMeans = observedSubjectMeans(subjectsIncluded, :, :);
residualSubjectMeans = residualSubjectMeans(subjectsIncluded, :, :);
observedPatternSubjectMeans = ...
    observedPatternSubjectMeans(subjectsIncluded, :, :);
nSubjects = sum(subjectsIncluded);

assert(nSubjects > 0, 'No participants could be included in the RT PPC.')

replicatedGroupMeans = divideReplicatedSums( ...
    replicatedSequenceSum, replicatedSequenceSubjectCount);
replicatedPatternGroupMeans = divideReplicatedSums( ...
    replicatedPatternSum, replicatedPatternSubjectCount);

observedGroupMeans = squeeze(mean( ...
    observedSubjectMeans, 1, 'omitnan'));
residualGroupMeans = squeeze(mean( ...
    residualSubjectMeans, 1, 'omitnan'));
residualGroupSEM = groupSEM(residualSubjectMeans);
observedPatternGroupMeans = squeeze(mean( ...
    observedPatternSubjectMeans, 1, 'omitnan'));

[predictedGroupMeans, predictiveLower, predictiveUpper] = ...
    summarizeReplications(replicatedGroupMeans);
[predictedPatternMeans, patternLower, patternUpper] = ...
    summarizeReplications(replicatedPatternGroupMeans);

fitMetrics = calculateFitMetrics(observedGroupMeans, ...
    predictedGroupMeans, predictiveLower, predictiveUpper, ...
    conditionLabels);

[observedViolationCost, replicatedViolationCost, ...
    predictedViolationCost, violationCostLower, violationCostUpper] = ...
    calculateViolationCosts(observedPatternGroupMeans, ...
    replicatedPatternGroupMeans);

%% Plot 1: observed and replicated target-sequence curves

sequenceFigure = plotPredictiveSequenceCurves( ...
    observedGroupMeans, predictedGroupMeans, ...
    predictiveLower, predictiveUpper, sequenceLabels, ...
    conditionLabels, conditionColors, settings.figureVisible);

%% Plot 2: observed versus predicted sequence-cell means

scatterFigure = plotObservedVersusPredicted( ...
    observedGroupMeans, predictedGroupMeans, ...
    predictiveLower, predictiveUpper, conditionLabels, ...
    conditionColors, fitMetrics, settings.figureVisible);

%% Plot 3: remaining sequence effects in fitted residuals

residualFigure = plotResidualSequenceEffects( ...
    residualGroupMeans, residualGroupSEM, sequenceLabels, ...
    conditionLabels, conditionColors, settings.figureVisible);

%% Plot 4: pattern confirmation and violation

patternFigure = plotPatternConfirmation( ...
    observedPatternGroupMeans, predictedPatternMeans, ...
    patternLower, patternUpper, patternLabels, ...
    conditionLabels, conditionColors, settings.figureVisible);

violationFigure = plotViolationCosts( ...
    observedViolationCost, predictedViolationCost, ...
    violationCostLower, violationCostUpper, ...
    conditionLabels, conditionColors, settings.figureVisible);

%% Save figures and values

if settings.saveFigures
    saveFigurePDF(sequenceFigure, fullfile(outputDir, ...
        [outputStem '_SequenceCurves.pdf']))
    saveFigurePDF(scatterFigure, fullfile(outputDir, ...
        [outputStem '_ObservedVsPredicted.pdf']))
    saveFigurePDF(residualFigure, fullfile(outputDir, ...
        [outputStem '_ResidualSequenceEffects.pdf']))
    saveFigurePDF(patternFigure, fullfile(outputDir, ...
        [outputStem '_PatternConfirmation.pdf']))
    saveFigurePDF(violationFigure, fullfile(outputDir, ...
        [outputStem '_ViolationCost.pdf']))
end

ppcSummary = struct;
ppcSummary.settings = settings;
ppcSummary.ppcConfig = ppcConfig;
ppcSummary.subjectNames = subjectNames;
ppcSummary.sequenceLabels = sequenceLabels;
ppcSummary.conditionLabels = conditionLabels;
ppcSummary.patternLabels = patternLabels;
ppcSummary.observedSubjectMeans = observedSubjectMeans;
ppcSummary.observedGroupMeans = observedGroupMeans;
ppcSummary.replicatedGroupMeans = replicatedGroupMeans;
ppcSummary.predictedGroupMeans = predictedGroupMeans;
ppcSummary.predictiveLower = predictiveLower;
ppcSummary.predictiveUpper = predictiveUpper;
ppcSummary.residualSubjectMeans = residualSubjectMeans;
ppcSummary.residualGroupMeans = residualGroupMeans;
ppcSummary.residualGroupSEM = residualGroupSEM;
ppcSummary.observedPatternSubjectMeans = ...
    observedPatternSubjectMeans;
ppcSummary.observedPatternGroupMeans = observedPatternGroupMeans;
ppcSummary.replicatedPatternGroupMeans = replicatedPatternGroupMeans;
ppcSummary.predictedPatternMeans = predictedPatternMeans;
ppcSummary.patternLower = patternLower;
ppcSummary.patternUpper = patternUpper;
ppcSummary.observedViolationCost = observedViolationCost;
ppcSummary.replicatedViolationCost = replicatedViolationCost;
ppcSummary.predictedViolationCost = predictedViolationCost;
ppcSummary.violationCostLower = violationCostLower;
ppcSummary.violationCostUpper = violationCostUpper;
ppcSummary.fitMetrics = fitMetrics;
ppcSummary.replicatedSequenceSubjectCount = ...
    replicatedSequenceSubjectCount;
ppcSummary.replicatedPatternSubjectCount = ...
    replicatedPatternSubjectCount;

save(fullfile(outputDir, [outputStem '_values.mat']), ...
    'ppcSummary', '-v7.3')

fprintf('\nIncluded %d participants and %d replicated datasets.\n', ...
    nSubjects, settings.nReplications)
disp(fitMetrics)
fprintf('Saved RT predictive checks to:\n%s\n', outputDir)

%% Local functions

function [labels, powers] = makeSequenceLabels(historyLength)

nSequences = 2 ^ historyLength;
labels = strings(nSequences, 1);
powers = 2 .^ (historyLength - 1:-1:0);

for sequence = 0:nSequences - 1
    binarySequence = dec2bin(sequence, historyLength);
    textSequence = strings(1, historyLength);
    textSequence(binarySequence == '0') = "L";
    textSequence(binarySequence == '1') = "R";
    labels(sequence + 1) = strjoin(textSequence, '');
end
end

function [sequenceCode, currentTrial] = encodeSequenceThroughCurrent( ...
    historyData, historyLength, powers)

currentTrial = (historyLength + 1:numel(historyData))';
sequenceCode = zeros(numel(currentTrial), 1);
validHistory = true(numel(currentTrial), 1);

nSequencePositions = historyLength + 1;
for position = 1:nSequencePositions
    historyTrial = currentTrial - position + 1;
    value = historyData(historyTrial);
    validHistory = validHistory & ismember(value, [-1 1]);
    sequenceCode = sequenceCode + ...
        (value == 1) .* powers(position);
end

sequenceCode(~validHistory) = nan;
end

function patternCode = encodePatternConfirmation(target, currentTrial)

currentRepeat = target(currentTrial) == target(currentTrial - 1);
previousRepeat = ...
    target(currentTrial - 1) == target(currentTrial - 2);
earlierRepeat = ...
    target(currentTrial - 2) == target(currentTrial - 3);

patternCode = nan(size(currentTrial));
patternCode(previousRepeat & earlierRepeat & currentRepeat) = 1;
patternCode(previousRepeat & earlierRepeat & ~currentRepeat) = 2;
patternCode(~previousRepeat & ~earlierRepeat & ~currentRepeat) = 3;
patternCode(~previousRepeat & ~earlierRepeat & currentRepeat) = 4;
end

function simulatedRT = simulateLogNormalRT(mu, sigma, nReplications, ...
    minimumRT, maximumRT, resampleOutsideRange)

simulatedLogRT = mu + sigma .* randn(numel(mu), nReplications);
simulatedRT = exp(simulatedLogRT);

if ~resampleOutsideRange
    return
end

outsideRange = simulatedRT <= minimumRT | simulatedRT >= maximumRT;
muMatrix = repmat(mu, 1, nReplications);
while any(outsideRange(:))
    simulatedLogRT(outsideRange) = ...
        muMatrix(outsideRange) + ...
        sigma .* randn(sum(outsideRange(:)), 1);
    simulatedRT(outsideRange) = exp(simulatedLogRT(outsideRange));
    outsideRange = simulatedRT <= minimumRT | simulatedRT >= maximumRT;
end
end

function [minimumRT, maximumRT] = subjectRTRange(subjectFit)

minimumRT = 0;
maximumRT = inf;
if isfield(subjectFit, 'settings')
    if isfield(subjectFit.settings, 'minRT')
        minimumRT = subjectFit.settings.minRT;
    end
    if isfield(subjectFit.settings, 'maxRT')
        maximumRT = subjectFit.settings.maxRT;
    end
end
end

function groupValues = divideReplicatedSums(replicatedSum, subjectCount)

groupValues = nan(size(replicatedSum));
for condition = 1:size(replicatedSum, 3)
    for cellIndex = 1:size(replicatedSum, 2)
        if subjectCount(cellIndex, condition) > 0
            groupValues(:, cellIndex, condition) = ...
                replicatedSum(:, cellIndex, condition) ./ ...
                subjectCount(cellIndex, condition);
        end
    end
end
end

function sem = groupSEM(subjectValues)

nCells = size(subjectValues, 2);
nConditions = size(subjectValues, 3);
sem = nan(nCells, nConditions);

for condition = 1:nConditions
    for cellIndex = 1:nCells
        values = subjectValues(:, cellIndex, condition);
        values = values(isfinite(values));
        if numel(values) > 1
            sem(cellIndex, condition) = ...
                std(values) / sqrt(numel(values));
        end
    end
end
end

function [predictedMean, lowerBound, upperBound] = ...
    summarizeReplications(replicatedValues)

nCells = size(replicatedValues, 2);
nConditions = size(replicatedValues, 3);
predictedMean = nan(nCells, nConditions);
lowerBound = nan(nCells, nConditions);
upperBound = nan(nCells, nConditions);

for condition = 1:nConditions
    for cellIndex = 1:nCells
        values = replicatedValues(:, cellIndex, condition);
        values = values(isfinite(values));
        if ~isempty(values)
            predictedMean(cellIndex, condition) = mean(values);
            bounds = prctile(values, [2.5 97.5]);
            lowerBound(cellIndex, condition) = bounds(1);
            upperBound(cellIndex, condition) = bounds(2);
        end
    end
end
end

function metrics = calculateFitMetrics(observed, predicted, ...
    lowerBound, upperBound, conditionLabels)

nConditions = size(observed, 2);
correlation = nan(nConditions, 1);
RMSE = nan(nConditions, 1);
coveragePercent = nan(nConditions, 1);
nCells = zeros(nConditions, 1);

for condition = 1:nConditions
    valid = isfinite(observed(:, condition)) & ...
        isfinite(predicted(:, condition));
    nCells(condition) = sum(valid);

    if sum(valid) > 1
        correlationMatrix = corrcoef( ...
            observed(valid, condition), predicted(valid, condition));
        correlation(condition) = correlationMatrix(1, 2);
    end

    if any(valid)
        difference = ...
            observed(valid, condition) - predicted(valid, condition);
        RMSE(condition) = sqrt(mean(difference .^ 2));
        covered = observed(valid, condition) >= ...
            lowerBound(valid, condition) & ...
            observed(valid, condition) <= ...
            upperBound(valid, condition);
        coveragePercent(condition) = mean(covered) * 100;
    end
end

metrics = table(string(conditionLabels(:)), nCells, correlation, ...
    RMSE, coveragePercent, 'VariableNames', ...
    {'Condition', 'NCells', 'Correlation', 'RMSE_ms', ...
    'PredictiveCoveragePercent'});
end

function [observedCost, replicatedCost, predictedCost, ...
    lowerBound, upperBound] = calculateViolationCosts( ...
    observedPatternMeans, replicatedPatternMeans)

% Patterns 1 and 3 confirm an established pattern; 2 and 4 violate one.
observedConfirmation = mean(observedPatternMeans([1 3], :), 1, 'omitnan');
observedViolation = mean(observedPatternMeans([2 4], :), 1, 'omitnan');
observedCost = observedViolation - observedConfirmation;

replicatedConfirmation = mean( ...
    replicatedPatternMeans(:, [1 3], :), 2, 'omitnan');
replicatedViolation = mean( ...
    replicatedPatternMeans(:, [2 4], :), 2, 'omitnan');
replicatedCost = squeeze( ...
    replicatedViolation - replicatedConfirmation);

nConditions = size(observedPatternMeans, 2);
predictedCost = nan(1, nConditions);
lowerBound = nan(1, nConditions);
upperBound = nan(1, nConditions);
for condition = 1:nConditions
    values = replicatedCost(:, condition);
    values = values(isfinite(values));
    predictedCost(condition) = mean(values);
    bounds = prctile(values, [2.5 97.5]);
    lowerBound(condition) = bounds(1);
    upperBound(condition) = bounds(2);
end
end

function figureHandle = plotPredictiveSequenceCurves( ...
    observed, predicted, lowerBound, upperBound, sequenceLabels, ...
    conditionLabels, colors, figureVisible)

nSequences = size(observed, 1);
x = 1:nSequences;
figureHandle = figure('Color', 'w', 'Visible', figureVisible, ...
    'Position', [50 80 1550 620]);
layout = tiledlayout(figureHandle, 1, 2, ...
    'TileSpacing', 'compact', 'Padding', 'compact');

for condition = 1:2
    axesHandle = nexttile(layout);
    hold(axesHandle, 'on')

    intervalHandle = fill(axesHandle, ...
        [x fliplr(x)], ...
        [lowerBound(:, condition)' fliplr(upperBound(:, condition)')], ...
        colors(condition, :), 'EdgeColor', 'none', 'FaceAlpha', 0.18);
    predictedHandle = plot(axesHandle, x, predicted(:, condition), ...
        '-', 'Color', colors(condition, :), 'LineWidth', 2.4);
    observedHandle = plot(axesHandle, x, observed(:, condition), ...
        'ko-', 'MarkerFaceColor', 'w', 'MarkerSize', 5, ...
        'LineWidth', 1.35);

    xticks(axesHandle, x)
    xticklabels(axesHandle, sequenceLabels)
    xtickangle(axesHandle, 45)
    xlim(axesHandle, [0.5 nSequences + 0.5])
    xlabel(axesHandle, ...
        'Target sequence (first letter is current trial)')
    ylabel(axesHandle, 'Mean RT (ms)')
    title(axesHandle, conditionLabels{condition}, ...
        'FontWeight', 'normal')
    box(axesHandle, 'off')
    set(axesHandle, 'FontSize', 12, 'TickDir', 'out')

    if condition == 1
        legend(axesHandle, ...
            [observedHandle predictedHandle intervalHandle], ...
            {'Observed', 'Model mean', '95% predictive interval'}, ...
            'Location', 'best', 'Box', 'off')
    end
end

title(layout, ...
    'Target-sequence RT effects: observed and model-replicated data', ...
    'FontWeight', 'normal')
end

function figureHandle = plotObservedVersusPredicted( ...
    observed, predicted, lowerBound, upperBound, conditionLabels, ...
    colors, fitMetrics, figureVisible)

allFiniteValues = [observed(:); lowerBound(:); upperBound(:)];
allFiniteValues = allFiniteValues(isfinite(allFiniteValues));
plotMinimum = min(allFiniteValues);
plotMaximum = max(allFiniteValues);
plotPadding = max(5, 0.05 * (plotMaximum - plotMinimum));
plotLimits = [plotMinimum - plotPadding, plotMaximum + plotPadding];

figureHandle = figure('Color', 'w', 'Visible', figureVisible, ...
    'Position', [100 100 720 650]);
axesHandle = axes(figureHandle);
hold(axesHandle, 'on')
plot(axesHandle, plotLimits, plotLimits, '--', ...
    'Color', [0.45 0.45 0.45], 'LineWidth', 1.3)

scatterHandles = gobjects(2, 1);
for condition = 1:2
    valid = isfinite(observed(:, condition)) & ...
        isfinite(predicted(:, condition));
    lowerError = predicted(valid, condition) - ...
        lowerBound(valid, condition);
    upperError = upperBound(valid, condition) - ...
        predicted(valid, condition);

    errorbar(axesHandle, observed(valid, condition), ...
        predicted(valid, condition), lowerError, upperError, ...
        '.', 'Color', colors(condition, :), ...
        'LineWidth', 1, 'CapSize', 5);
    scatterHandles(condition) = scatter(axesHandle, ...
        observed(valid, condition), predicted(valid, condition), ...
        55, colors(condition, :), 'filled', ...
        'MarkerEdgeColor', 'w', 'LineWidth', 0.7);
end

xlim(axesHandle, plotLimits)
ylim(axesHandle, plotLimits)
axis(axesHandle, 'square')
xlabel(axesHandle, 'Observed sequence-cell mean RT (ms)')
ylabel(axesHandle, 'Model-predicted sequence-cell mean RT (ms)')
title(axesHandle, 'Observed versus model-predicted sequence effects', ...
    'FontWeight', 'normal')
legend(axesHandle, scatterHandles, conditionLabels, ...
    'Location', 'best', 'Box', 'off')
box(axesHandle, 'off')
set(axesHandle, 'FontSize', 13, 'TickDir', 'out')

annotationText = sprintf( ...
    ['Congruent: r = %.2f, RMSE = %.1f ms, coverage = %.0f%%\n' ...
     'Incongruent: r = %.2f, RMSE = %.1f ms, coverage = %.0f%%'], ...
    fitMetrics.Correlation(1), fitMetrics.RMSE_ms(1), ...
    fitMetrics.PredictiveCoveragePercent(1), ...
    fitMetrics.Correlation(2), fitMetrics.RMSE_ms(2), ...
    fitMetrics.PredictiveCoveragePercent(2));
text(axesHandle, 0.04, 0.96, annotationText, 'Units', 'normalized', ...
    'VerticalAlignment', 'top', 'FontSize', 11, ...
    'BackgroundColor', 'w', 'Margin', 6)
end

function figureHandle = plotResidualSequenceEffects( ...
    residualMean, residualSEM, sequenceLabels, ...
    conditionLabels, colors, figureVisible)

nSequences = size(residualMean, 1);
x = 1:nSequences;
figureHandle = figure('Color', 'w', 'Visible', figureVisible, ...
    'Position', [50 80 1550 620]);
layout = tiledlayout(figureHandle, 1, 2, ...
    'TileSpacing', 'compact', 'Padding', 'compact');

for condition = 1:2
    axesHandle = nexttile(layout);
    hold(axesHandle, 'on')
    yline(axesHandle, 0, '--', 'Color', [0.45 0.45 0.45], ...
        'LineWidth', 1.2);
    errorbar(axesHandle, x, residualMean(:, condition), ...
        residualSEM(:, condition), 'o-', ...
        'Color', colors(condition, :), ...
        'MarkerFaceColor', colors(condition, :), ...
        'LineWidth', 1.8, 'CapSize', 7);

    xticks(axesHandle, x)
    xticklabels(axesHandle, sequenceLabels)
    xtickangle(axesHandle, 45)
    xlim(axesHandle, [0.5 nSequences + 0.5])
    xlabel(axesHandle, ...
        'Target sequence (first letter is current trial)')
    ylabel(axesHandle, 'Mean residual log(RT)')
    title(axesHandle, conditionLabels{condition}, ...
        'FontWeight', 'normal')
    box(axesHandle, 'off')
    set(axesHandle, 'FontSize', 12, 'TickDir', 'out')
end

title(layout, ...
    'Remaining target-sequence structure after fitting the DBM', ...
    'FontWeight', 'normal')
end

function figureHandle = plotPatternConfirmation( ...
    observed, predicted, lowerBound, upperBound, patternLabels, ...
    conditionLabels, colors, figureVisible)

nPatterns = size(observed, 1);
x = 1:nPatterns;
figureHandle = figure('Color', 'w', 'Visible', figureVisible, ...
    'Position', [80 100 1200 580]);
layout = tiledlayout(figureHandle, 1, 2, ...
    'TileSpacing', 'compact', 'Padding', 'compact');

for condition = 1:2
    axesHandle = nexttile(layout);
    hold(axesHandle, 'on')
    intervalHandle = fill(axesHandle, ...
        [x fliplr(x)], ...
        [lowerBound(:, condition)' fliplr(upperBound(:, condition)')], ...
        colors(condition, :), 'EdgeColor', 'none', 'FaceAlpha', 0.18);
    predictedHandle = plot(axesHandle, x, predicted(:, condition), ...
        '-', 'Color', colors(condition, :), 'LineWidth', 2.4);
    observedHandle = plot(axesHandle, x, observed(:, condition), ...
        'ko-', 'MarkerFaceColor', 'w', 'MarkerSize', 6, ...
        'LineWidth', 1.4);

    xticks(axesHandle, x)
    xticklabels(axesHandle, patternLabels)
    xlim(axesHandle, [0.6 nPatterns + 0.4])
    ylabel(axesHandle, 'Mean RT (ms)')
    title(axesHandle, conditionLabels{condition}, ...
        'FontWeight', 'normal')
    box(axesHandle, 'off')
    set(axesHandle, 'FontSize', 12, 'TickDir', 'out')

    if condition == 1
        legend(axesHandle, ...
            [observedHandle predictedHandle intervalHandle], ...
            {'Observed', 'Model mean', '95% predictive interval'}, ...
            'Location', 'best', 'Box', 'off')
    end
end

title(layout, ...
    'Continuation and violation of established transition patterns', ...
    'FontWeight', 'normal')
end

function figureHandle = plotViolationCosts( ...
    observed, predicted, lowerBound, upperBound, ...
    conditionLabels, colors, figureVisible)

x = 1:numel(observed);
figureHandle = figure('Color', 'w', 'Visible', figureVisible, ...
    'Position', [150 120 700 580]);
axesHandle = axes(figureHandle);
hold(axesHandle, 'on')
yline(axesHandle, 0, '--', 'Color', [0.45 0.45 0.45], ...
    'LineWidth', 1.2);

barHandles = gobjects(numel(observed), 1);
modelHandle = gobjects(1);
for condition = 1:numel(observed)
    barHandles(condition) = bar(axesHandle, x(condition), ...
        observed(condition), 0.55, ...
        'FaceColor', colors(condition, :), 'FaceAlpha', 0.75);
    thisModelHandle = errorbar(axesHandle, ...
        x(condition), predicted(condition), ...
        predicted(condition) - lowerBound(condition), ...
        upperBound(condition) - predicted(condition), ...
        'kd', 'MarkerFaceColor', 'w', 'MarkerSize', 7, ...
        'LineWidth', 1.5, 'CapSize', 10);
    if condition == 1
        modelHandle = thisModelHandle;
    end
end

xticks(axesHandle, x)
xticklabels(axesHandle, conditionLabels)
ylabel(axesHandle, 'Violation cost: RT_{violate} - RT_{confirm} (ms)')
title(axesHandle, ...
    'Observed violation cost and model predictive interval', ...
    'FontWeight', 'normal')
legend(axesHandle, ...
    [barHandles(1), modelHandle], ...
    {'Observed', 'Model mean and 95% interval'}, ...
    'Location', 'best', 'Box', 'off')
box(axesHandle, 'off')
set(axesHandle, 'FontSize', 13, 'TickDir', 'out')
end

function saveFigurePDF(figureHandle, outputFile)

if exist('exportgraphics', 'file')
    exportgraphics(figureHandle, outputFile, 'ContentType', 'vector')
else
    print(figureHandle, erase(outputFile, '.pdf'), '-dpdf', '-painters')
end
end

function assertFieldsPresent(dataStructure, requiredFields, fileName)

missingFields = requiredFields(~isfield(dataStructure, requiredFields));
if ~isempty(missingFields)
    error('Missing field(s) in %s: %s', fileName, ...
        strjoin(missingFields, ', '))
end
end
