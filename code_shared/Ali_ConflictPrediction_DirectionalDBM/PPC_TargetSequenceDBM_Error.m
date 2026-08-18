%% Parametric predictive checks for the target-sequence DBM error model
%
% This script:
%   1) Loads completed subject-level fits from DBM_TargetSequence_Error.m
%   2) Simulates replicated errors from each fitted Bernoulli model
%   3) Applies the same t0:t-3 target-sequence bins used by
%      PPC_TargetSequenceDBM_RT.m and PlotSequenceEffects_RT_pHand.m
%   4) Compares observed error rates with predictive distributions
%   5) Plots remaining sequence effects and pattern-violation contrasts
%
% Sequence labels are ordered current target first: t0, t-1, t-2, t-3.
% Error rates and residuals are shown in percentage points.
%
% The predictive intervals condition on the fitted parameter estimates.
% They include Bernoulli observation noise, but not uncertainty in alpha
% or the logistic-regression coefficients. They are therefore parametric
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
    'TargetSequenceDBM', 'Error', 'SubjectData');
outputDir = fullfile(rootDir, 'Figures', ...
    'TargetSequenceDBM', 'Error_PPC');
fitPattern = '*_TargetSequenceDBM_Error.mat';
outputStem = 'PPC_TargetSequenceDBM_Error';

if isfield(ppcConfig, 'fitDir'), fitDir = ppcConfig.fitDir; end
if isfield(ppcConfig, 'outputDir'), outputDir = ppcConfig.outputDir; end
if isfield(ppcConfig, 'fitPattern'), fitPattern = ppcConfig.fitPattern; end
if isfield(ppcConfig, 'outputStem'), outputStem = ppcConfig.outputStem; end

assert(isfolder(dataDir), 'Behavioral data directory was not found.')
assert(isfolder(fitDir), 'Error DBM subject-fit directory was not found.')
if ~isfolder(outputDir)
    mkdir(outputDir)
end

rng(settings.randomSeed, 'twister')

%% Find completed fits and prepare sequence definitions

fitFiles = dir(fullfile(fitDir, fitPattern));
assert(~isempty(fitFiles), 'No completed subject-level error fits were found.')
[~, fitOrder] = sort({fitFiles.name});
fitFiles = fitFiles(fitOrder);
fitFiles = fitFiles( ...
    1:min(settings.nSubjectsRequested, numel(fitFiles)));

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

observedSubjectRates = ...
    nan(nSubjectsAvailable, nSequences, nConditions);
expectedSubjectRates = ...
    nan(nSubjectsAvailable, nSequences, nConditions);
residualSubjectRates = ...
    nan(nSubjectsAvailable, nSequences, nConditions);
observedPatternSubjectRates = ...
    nan(nSubjectsAvailable, nPatterns, nConditions);
expectedPatternSubjectRates = ...
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
        fprintf('Error predictive check: subject %d/%d\n', ...
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

    requiredFields = {'Target', 'Error', 'Congruence'};
    assertFieldsPresent(allData, requiredFields, behavioralFile)

    requiredFitFields = {'predictedError', 'validError', 'blockStart'};
    if ~all(isfield(subjectFit, requiredFitFields))
        warning('Required fitted values were not found for %s; skipping.', ...
            subjectName)
        continue
    end

    nTrials = min([numel(allData.Target), numel(allData.Error), ...
        numel(allData.Congruence), ...
        numel(subjectFit.predictedError), ...
        numel(subjectFit.validError), ...
        numel(subjectFit.blockStart)]);

    if nTrials <= settings.historyLength
        warning('Insufficient fitted data for %s; skipping.', subjectName)
        continue
    end

    target = double(allData.Target(1:nTrials));
    target = target(:);
    observedError = double(allData.Error(1:nTrials));
    observedError = observedError(:);
    congruence = double(allData.Congruence(1:nTrials));
    congruence = congruence(:);
    predictedError = double(subjectFit.predictedError(1:nTrials));
    predictedError = predictedError(:);
    modelValid = logical(subjectFit.validError(1:nTrials));
    modelValid = modelValid(:);
    blockStart = logical(subjectFit.blockStart(1:nTrials));
    blockStart = blockStart(:);

    [sequenceCode, currentTrial] = encodeSequenceThroughCurrent( ...
        target, settings.historyLength, sequencePowers);

    currentTarget = target(currentTrial);
    currentError = observedError(currentTrial);
    currentCongruence = congruence(currentTrial);
    currentProbability = predictedError(currentTrial);
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
        isfinite(currentError) & ismember(currentError, [0 1]) & ...
        isfinite(currentCongruence) & ...
        isfinite(currentProbability) & ...
        currentProbability >= 0 & currentProbability <= 1;

    if ~any(baseValid)
        warning('No valid predictive-check trials for %s; skipping.', ...
            subjectName)
        continue
    end

    simulatedError = false( ...
        numel(currentTrial), settings.nReplications);
    simulatedError(baseValid, :) = ...
        rand(sum(baseValid), settings.nReplications) < ...
        currentProbability(baseValid);

    rawResidual = currentError - currentProbability;
    patternCode = encodePatternConfirmation(target, currentTrial);

    for condition = 1:nConditions
        congruenceCode = 2 * condition - 3;

        for sequence = 1:nSequences
            useTrials = baseValid & ...
                sequenceCode == (sequence - 1) & ...
                currentCongruence == congruenceCode;

            if any(useTrials)
                observedSubjectRates(subjectIndex, sequence, condition) = ...
                    mean(currentError(useTrials)) * 100;
                expectedSubjectRates(subjectIndex, sequence, condition) = ...
                    mean(currentProbability(useTrials)) * 100;
                residualSubjectRates(subjectIndex, sequence, condition) = ...
                    mean(rawResidual(useTrials)) * 100;

                simulatedSubjectRate = ...
                    mean(simulatedError(useTrials, :), 1)' * 100;
                replicatedSequenceSum(:, sequence, condition) = ...
                    replicatedSequenceSum(:, sequence, condition) + ...
                    simulatedSubjectRate;
                replicatedSequenceSubjectCount(sequence, condition) = ...
                    replicatedSequenceSubjectCount(sequence, condition) + 1;
            end
        end

        for pattern = 1:nPatterns
            useTrials = baseValid & patternCode == pattern & ...
                currentCongruence == congruenceCode;

            if any(useTrials)
                observedPatternSubjectRates( ...
                    subjectIndex, pattern, condition) = ...
                    mean(currentError(useTrials)) * 100;
                expectedPatternSubjectRates( ...
                    subjectIndex, pattern, condition) = ...
                    mean(currentProbability(useTrials)) * 100;

                simulatedSubjectRate = ...
                    mean(simulatedError(useTrials, :), 1)' * 100;
                replicatedPatternSum(:, pattern, condition) = ...
                    replicatedPatternSum(:, pattern, condition) + ...
                    simulatedSubjectRate;
                replicatedPatternSubjectCount(pattern, condition) = ...
                    replicatedPatternSubjectCount(pattern, condition) + 1;
            end
        end
    end

    subjectsIncluded(subjectIndex) = true;
end

%% Group summaries and predictive intervals

subjectNames = subjectNames(subjectsIncluded);
observedSubjectRates = observedSubjectRates(subjectsIncluded, :, :);
expectedSubjectRates = expectedSubjectRates(subjectsIncluded, :, :);
residualSubjectRates = residualSubjectRates(subjectsIncluded, :, :);
observedPatternSubjectRates = ...
    observedPatternSubjectRates(subjectsIncluded, :, :);
expectedPatternSubjectRates = ...
    expectedPatternSubjectRates(subjectsIncluded, :, :);
nSubjects = sum(subjectsIncluded);

assert(nSubjects > 0, ...
    'No participants could be included in the error PPC.')

replicatedGroupRates = divideReplicatedSums( ...
    replicatedSequenceSum, replicatedSequenceSubjectCount);
replicatedPatternGroupRates = divideReplicatedSums( ...
    replicatedPatternSum, replicatedPatternSubjectCount);

observedGroupRates = squeeze(mean( ...
    observedSubjectRates, 1, 'omitnan'));
expectedGroupRates = squeeze(mean( ...
    expectedSubjectRates, 1, 'omitnan'));
residualGroupRates = squeeze(mean( ...
    residualSubjectRates, 1, 'omitnan'));
residualGroupSEM = groupSEM(residualSubjectRates);
observedPatternGroupRates = squeeze(mean( ...
    observedPatternSubjectRates, 1, 'omitnan'));
expectedPatternGroupRates = squeeze(mean( ...
    expectedPatternSubjectRates, 1, 'omitnan'));

[predictedGroupRates, predictiveLower, predictiveUpper] = ...
    summarizeReplications(replicatedGroupRates);
[predictedPatternRates, patternLower, patternUpper] = ...
    summarizeReplications(replicatedPatternGroupRates);

fitMetrics = calculateFitMetrics(observedGroupRates, ...
    predictedGroupRates, predictiveLower, predictiveUpper, ...
    conditionLabels);

[observedViolationCost, replicatedViolationCost, ...
    predictedViolationCost, violationCostLower, violationCostUpper] = ...
    calculateViolationCosts(observedPatternGroupRates, ...
    replicatedPatternGroupRates);

%% Plot predictive checks

sequenceFigure = plotPredictiveSequenceCurves( ...
    observedGroupRates, predictedGroupRates, ...
    predictiveLower, predictiveUpper, sequenceLabels, ...
    conditionLabels, conditionColors, settings.figureVisible);

scatterFigure = plotObservedVersusPredicted( ...
    observedGroupRates, predictedGroupRates, ...
    predictiveLower, predictiveUpper, conditionLabels, ...
    conditionColors, fitMetrics, settings.figureVisible);

residualFigure = plotResidualSequenceEffects( ...
    residualGroupRates, residualGroupSEM, sequenceLabels, ...
    conditionLabels, conditionColors, settings.figureVisible);

patternFigure = plotPatternConfirmation( ...
    observedPatternGroupRates, predictedPatternRates, ...
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
ppcSummary.observedSubjectRates = observedSubjectRates;
ppcSummary.expectedSubjectRates = expectedSubjectRates;
ppcSummary.observedGroupRates = observedGroupRates;
ppcSummary.expectedGroupRates = expectedGroupRates;
ppcSummary.replicatedGroupRates = replicatedGroupRates;
ppcSummary.predictedGroupRates = predictedGroupRates;
ppcSummary.predictiveLower = predictiveLower;
ppcSummary.predictiveUpper = predictiveUpper;
ppcSummary.residualSubjectRates = residualSubjectRates;
ppcSummary.residualGroupRates = residualGroupRates;
ppcSummary.residualGroupSEM = residualGroupSEM;
ppcSummary.observedPatternSubjectRates = ...
    observedPatternSubjectRates;
ppcSummary.expectedPatternSubjectRates = ...
    expectedPatternSubjectRates;
ppcSummary.observedPatternGroupRates = observedPatternGroupRates;
ppcSummary.expectedPatternGroupRates = expectedPatternGroupRates;
ppcSummary.replicatedPatternGroupRates = replicatedPatternGroupRates;
ppcSummary.predictedPatternRates = predictedPatternRates;
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

save(fullfile(outputDir, ...
    [outputStem '_values.mat']), ...
    'ppcSummary', '-v7.3')

fprintf('\nIncluded %d participants and %d replicated datasets.\n', ...
    nSubjects, settings.nReplications)
disp(fitMetrics)
fprintf('Saved error predictive checks to:\n%s\n', outputDir)

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
    {'Condition', 'NCells', 'Correlation', ...
    'RMSE_percentagePoints', 'PredictiveCoveragePercent'});
end

function [observedCost, replicatedCost, predictedCost, ...
    lowerBound, upperBound] = calculateViolationCosts( ...
    observedPatternRates, replicatedPatternRates)

% Patterns 1 and 3 confirm an established pattern; 2 and 4 violate one.
observedConfirmation = mean( ...
    observedPatternRates([1 3], :), 1, 'omitnan');
observedViolation = mean( ...
    observedPatternRates([2 4], :), 1, 'omitnan');
observedCost = observedViolation - observedConfirmation;

replicatedConfirmation = mean( ...
    replicatedPatternRates(:, [1 3], :), 2, 'omitnan');
replicatedViolation = mean( ...
    replicatedPatternRates(:, [2 4], :), 2, 'omitnan');
replicatedCost = squeeze( ...
    replicatedViolation - replicatedConfirmation);

nConditions = size(observedPatternRates, 2);
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
    ylabel(axesHandle, 'Error rate (%)')
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
    'Target-sequence error effects: observed and model-replicated data', ...
    'FontWeight', 'normal')
end

function figureHandle = plotObservedVersusPredicted( ...
    observed, predicted, lowerBound, upperBound, conditionLabels, ...
    colors, fitMetrics, figureVisible)

allFiniteValues = [observed(:); lowerBound(:); upperBound(:)];
allFiniteValues = allFiniteValues(isfinite(allFiniteValues));
plotMinimum = min(allFiniteValues);
plotMaximum = max(allFiniteValues);
plotPadding = max(0.5, 0.05 * (plotMaximum - plotMinimum));
plotLimits = [max(0, plotMinimum - plotPadding), ...
    plotMaximum + plotPadding];
if plotLimits(2) <= plotLimits(1)
    plotLimits(2) = plotLimits(1) + 1;
end

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
xlabel(axesHandle, 'Observed sequence-cell error rate (%)')
ylabel(axesHandle, 'Model-predicted sequence-cell error rate (%)')
title(axesHandle, 'Observed versus model-predicted sequence effects', ...
    'FontWeight', 'normal')
legend(axesHandle, scatterHandles, conditionLabels, ...
    'Location', 'southeast', 'Box', 'off')
box(axesHandle, 'off')
set(axesHandle, 'FontSize', 13, 'TickDir', 'out')

annotationText = sprintf( ...
    ['Congruent: r = %.2f, RMSE = %.2f pp, coverage = %.0f%%\n' ...
     'Incongruent: r = %.2f, RMSE = %.2f pp, coverage = %.0f%%'], ...
    fitMetrics.Correlation(1), ...
    fitMetrics.RMSE_percentagePoints(1), ...
    fitMetrics.PredictiveCoveragePercent(1), ...
    fitMetrics.Correlation(2), ...
    fitMetrics.RMSE_percentagePoints(2), ...
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
    ylabel(axesHandle, 'Observed - predicted error rate (pp)')
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
    ylabel(axesHandle, 'Error rate (%)')
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
    'Errors after continuation and violation of transition patterns', ...
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
ylabel(axesHandle, ...
    'Violation cost: error_{violate} - error_{confirm} (pp)')
title(axesHandle, ...
    'Observed violation cost and model predictive interval', ...
    'FontWeight', 'normal')
legend(axesHandle, ...
    [barHandles(1), modelHandle], ...
    {'Observed', 'Model mean and 95% interval'}, ...
    'Location', 'northwest', 'Box', 'off')
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
