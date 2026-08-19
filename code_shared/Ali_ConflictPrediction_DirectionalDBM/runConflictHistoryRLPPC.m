function ppcValues = runConflictHistoryRLPPC(outcomeType, settings)
%RUNCONFLICTHISTORYRLPPC Parametric predictive checks by conflict history.
%
% The checks use the 16 combinations of current congruence and the three
% preceding congruence values shown in the behavioral regression figures.
% Replicated RTs are log-normal and replicated errors are Bernoulli draws,
% conditional on each participant's fitted parameters.

if nargin < 1 || isempty(outcomeType)
    error('Specify outcomeType as ''RT'' or ''Error''.')
end
if nargin < 2 || isempty(settings), settings = struct; end
outcomeType = validatestring(outcomeType, {'RT', 'Error'});

rootDir = fileparts(mfilename('fullpath'));
if isempty(rootDir), rootDir = pwd; end
settings = applyDefaults(settings, rootDir, outcomeType);
if ~isfolder(settings.outputDir), mkdir(settings.outputDir); end
rng(settings.randomSeed, 'twister')

fitFiles = dir(fullfile(settings.fitDir, settings.fitPattern));
assert(~isempty(fitFiles), 'No completed %s participant fits were found.', outcomeType)
[~, order] = sort({fitFiles.name});
fitFiles = fitFiles(order);
fitFiles = fitFiles(1:min(settings.nSubjectsRequested, numel(fitFiles)));

nSubjectsAvailable = numel(fitFiles);
nSequences = 2 ^ (settings.historyLength + 1);
nHistoryCounts = settings.historyLength + 1;
nConditions = 2;
nRep = settings.nReplications;

[sequenceLabels, historyLabels, sequenceCurrent, sequenceHistoryCount] = ...
    makeConflictSequenceLabels(settings.historyLength);

observedSubjectCells = nan(nSubjectsAvailable, nSequences);
expectedSubjectCells = nan(nSubjectsAvailable, nSequences);
residualSubjectCells = nan(nSubjectsAvailable, nSequences);
observedSubjectCounts = nan(nSubjectsAvailable, nHistoryCounts, nConditions);
expectedSubjectCounts = nan(nSubjectsAvailable, nHistoryCounts, nConditions);
residualSubjectCounts = nan(nSubjectsAvailable, nHistoryCounts, nConditions);

replicatedCellSum = zeros(nRep, nSequences);
replicatedCellSubjectCount = zeros(1, nSequences);
replicatedCountSum = zeros(nRep, nHistoryCounts, nConditions);
replicatedCountSubjectCount = zeros(nHistoryCounts, nConditions);

subjectNames = strings(nSubjectsAvailable, 1);
subjectsIncluded = false(nSubjectsAvailable, 1);

for subjectIndex = 1:nSubjectsAvailable
    if subjectIndex == 1 || mod(subjectIndex, 25) == 0 || ...
            subjectIndex == nSubjectsAvailable
        fprintf('%s conflict-history PPC: subject %d/%d\n', ...
            outcomeType, subjectIndex, nSubjectsAvailable)
    end

    loadedFit = load(fullfile(settings.fitDir, fitFiles(subjectIndex).name), ...
        'subjectFit');
    if ~isfield(loadedFit, 'subjectFit'), continue; end
    subjectFit = loadedFit.subjectFit;
    subjectName = string(subjectFit.subjectName);
    subjectNames(subjectIndex) = subjectName;
    behavioralFile = fullfile(settings.dataDir, char(subjectName + ".mat"));
    if ~isfile(behavioralFile), continue; end
    loadedData = load(behavioralFile, 'all');
    if ~isfield(loadedData, 'all'), continue; end
    allData = loadedData.all;

    if strcmp(outcomeType, 'RT')
        requiredFitFields = {'predictedLogRT', 'residualLogRT', ...
            'validRT', 'blockStart'};
        requiredDataFields = {'Congruence', 'LinRT'};
    else
        requiredFitFields = {'predictedError', 'validError', 'blockStart'};
        requiredDataFields = {'Congruence', 'Error'};
    end
    if ~all(isfield(subjectFit, requiredFitFields)) || ...
            ~all(isfield(allData, requiredDataFields))
        continue
    end

    nTrials = numel(allData.Congruence);
    for fieldIndex = 1:numel(requiredDataFields)
        nTrials = min(nTrials, numel(allData.(requiredDataFields{fieldIndex})));
    end
    for fieldIndex = 1:numel(requiredFitFields)
        nTrials = min(nTrials, numel(subjectFit.(requiredFitFields{fieldIndex})));
    end
    if nTrials <= settings.historyLength, continue; end

    incongruent = (columnDouble(allData.Congruence, nTrials) + 1) / 2;
    blockStart = logical(subjectFit.blockStart(1:nTrials));
    blockStart = blockStart(:);
    currentTrial = (settings.historyLength + 1:nTrials)';
    [sequenceCode, historyCount] = encodeConflictHistory( ...
        incongruent, currentTrial, settings.historyLength);

    withinBlock = true(size(currentTrial));
    if settings.excludeCrossBlockHistories
        blockID = cumsum(blockStart);
        for lag = 1:settings.historyLength
            withinBlock = withinBlock & ...
                blockID(currentTrial) == blockID(currentTrial - lag);
        end
    end

    modelValid = logical(subjectFit.validTrials(1:nTrials));
    modelValid = modelValid(:);
    currentValid = modelValid(currentTrial);
    currentCongruence = incongruent(currentTrial);
    baseValid = currentValid & withinBlock & isfinite(sequenceCode) & ...
        isfinite(historyCount) & ismember(currentCongruence, [0 1]);

    if strcmp(outcomeType, 'RT')
        observed = columnDouble(allData.LinRT, nTrials);
        observed = observed(currentTrial);
        mu = columnDouble(subjectFit.predictedLogRT, nTrials);
        mu = mu(currentTrial);
        residual = columnDouble(subjectFit.residualLogRT, nTrials);
        residual = residual(currentTrial);
        sigmaIndex = find(strcmp(subjectFit.parameterNames, 'logSigma'), 1);
        if isempty(sigmaIndex), continue; end
        sigma = exp(subjectFit.fitParams(sigmaIndex));
        baseValid = baseValid & isfinite(observed) & isfinite(mu) & ...
            isfinite(residual) & isfinite(sigma);
        replicated = nan(numel(currentTrial), nRep);
        replicated(baseValid, :) = simulateRT(mu(baseValid), sigma, ...
            nRep, settings.minRT, settings.maxRT, ...
            settings.resampleOutsideRTRange);
        expected = exp(mu + 0.5 * sigma ^ 2);
    else
        observed = columnDouble(allData.Error, nTrials);
        observed = observed(currentTrial);
        expected = columnDouble(subjectFit.predictedError, nTrials);
        expected = expected(currentTrial);
        residual = (observed - expected) * 100;
        baseValid = baseValid & isfinite(observed) & ...
            ismember(observed, [0 1]) & isfinite(expected) & ...
            expected >= 0 & expected <= 1;
        replicated = nan(numel(currentTrial), nRep);
        replicated(baseValid, :) = rand(sum(baseValid), nRep) < ...
            expected(baseValid);
        observed = observed * 100;
        expected = expected * 100;
        replicated = replicated * 100;
    end

    if ~any(baseValid), continue; end

    for sequence = 0:nSequences - 1
        useTrials = baseValid & sequenceCode == sequence;
        if ~any(useTrials), continue; end
        cellIndex = sequence + 1;
        observedSubjectCells(subjectIndex, cellIndex) = ...
            mean(observed(useTrials));
        expectedSubjectCells(subjectIndex, cellIndex) = ...
            mean(expected(useTrials));
        residualSubjectCells(subjectIndex, cellIndex) = ...
            mean(residual(useTrials));
        simulatedMean = mean(replicated(useTrials, :), 1)';
        replicatedCellSum(:, cellIndex) = ...
            replicatedCellSum(:, cellIndex) + simulatedMean;
        replicatedCellSubjectCount(cellIndex) = ...
            replicatedCellSubjectCount(cellIndex) + 1;
    end

    for condition = 0:1
        for count = 0:settings.historyLength
            useTrials = baseValid & currentCongruence == condition & ...
                historyCount == count;
            if ~any(useTrials), continue; end
            observedSubjectCounts(subjectIndex, count + 1, condition + 1) = ...
                mean(observed(useTrials));
            expectedSubjectCounts(subjectIndex, count + 1, condition + 1) = ...
                mean(expected(useTrials));
            residualSubjectCounts(subjectIndex, count + 1, condition + 1) = ...
                mean(residual(useTrials));
            simulatedMean = mean(replicated(useTrials, :), 1)';
            replicatedCountSum(:, count + 1, condition + 1) = ...
                replicatedCountSum(:, count + 1, condition + 1) + simulatedMean;
            replicatedCountSubjectCount(count + 1, condition + 1) = ...
                replicatedCountSubjectCount(count + 1, condition + 1) + 1;
        end
    end

    subjectsIncluded(subjectIndex) = true;
end

assert(any(subjectsIncluded), 'No participants could be included in the PPC.')
subjectNames = subjectNames(subjectsIncluded);
observedSubjectCells = observedSubjectCells(subjectsIncluded, :);
expectedSubjectCells = expectedSubjectCells(subjectsIncluded, :);
residualSubjectCells = residualSubjectCells(subjectsIncluded, :);
observedSubjectCounts = observedSubjectCounts(subjectsIncluded, :, :);
expectedSubjectCounts = expectedSubjectCounts(subjectsIncluded, :, :);
residualSubjectCounts = residualSubjectCounts(subjectsIncluded, :, :);
nSubjects = sum(subjectsIncluded);

replicatedGroupCells = divideReplicated(replicatedCellSum, ...
    replicatedCellSubjectCount);
replicatedGroupCounts = divideReplicatedCounts(replicatedCountSum, ...
    replicatedCountSubjectCount);

observedGroupCells = mean(observedSubjectCells, 1, 'omitnan');
expectedGroupCells = mean(expectedSubjectCells, 1, 'omitnan');
residualGroupCells = mean(residualSubjectCells, 1, 'omitnan');
observedGroupCounts = squeeze(mean(observedSubjectCounts, 1, 'omitnan'));
expectedGroupCounts = squeeze(mean(expectedSubjectCounts, 1, 'omitnan'));
residualGroupCounts = squeeze(mean(residualSubjectCounts, 1, 'omitnan'));

predictedMedianCells = median(replicatedGroupCells, 1, 'omitnan');
predictedCICells = prctile(replicatedGroupCells, [2.5 97.5], 1);
predictedMedianCounts = squeeze(median(replicatedGroupCounts, 1, 'omitnan'));
predictedCICounts = squeeze(prctile(replicatedGroupCounts, [2.5 97.5], 1));

conditionNames = {'Congruent', 'Incongruent'};
metrics = table('Size', [nConditions, 5], ...
    'VariableTypes', {'string', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'CurrentCongruence', 'Correlation', 'RMSE', ...
    'PredictiveCoverage', 'NCells'});
for condition = 0:1
    useCells = sequenceCurrent == condition;
    observedValues = observedGroupCells(useCells)';
    predictedValues = predictedMedianCells(useCells)';
    lower = predictedCICells(1, useCells)';
    upper = predictedCICells(2, useCells)';
    finite = isfinite(observedValues) & isfinite(predictedValues);
    metrics.CurrentCongruence(condition + 1) = conditionNames{condition + 1};
    metrics.Correlation(condition + 1) = ...
        corr(observedValues(finite), predictedValues(finite));
    metrics.RMSE(condition + 1) = ...
        sqrt(mean((observedValues(finite) - predictedValues(finite)) .^ 2));
    metrics.PredictiveCoverage(condition + 1) = ...
        mean(observedValues(finite) >= lower(finite) & ...
        observedValues(finite) <= upper(finite)) * 100;
    metrics.NCells(condition + 1) = sum(finite);
end

ppcValues = struct;
ppcValues.outcomeType = outcomeType;
ppcValues.settings = settings;
ppcValues.subjectNames = subjectNames;
ppcValues.sequenceLabels = sequenceLabels;
ppcValues.historyLabels = historyLabels;
ppcValues.sequenceCurrent = sequenceCurrent;
ppcValues.sequenceHistoryCount = sequenceHistoryCount;
ppcValues.observedSubjectCells = observedSubjectCells;
ppcValues.expectedSubjectCells = expectedSubjectCells;
ppcValues.residualSubjectCells = residualSubjectCells;
ppcValues.observedGroupCells = observedGroupCells;
ppcValues.expectedGroupCells = expectedGroupCells;
ppcValues.residualGroupCells = residualGroupCells;
ppcValues.replicatedGroupCells = replicatedGroupCells;
ppcValues.predictedMedianCells = predictedMedianCells;
ppcValues.predictedCICells = predictedCICells;
ppcValues.observedGroupCounts = observedGroupCounts;
ppcValues.expectedGroupCounts = expectedGroupCounts;
ppcValues.residualGroupCounts = residualGroupCounts;
ppcValues.replicatedGroupCounts = replicatedGroupCounts;
ppcValues.predictedMedianCounts = predictedMedianCounts;
ppcValues.predictedCICounts = predictedCICounts;
ppcValues.metrics = metrics;

outputStem = settings.outputStem;
save(fullfile(settings.outputDir, [outputStem '_values.mat']), ...
    'ppcValues', '-v7.3')
writetable(metrics, fullfile(settings.outputDir, [outputStem '_metrics.csv']))

makeHistoryCellFigure(ppcValues, settings, outputStem, nSubjects);
makeConflictCountFigure(ppcValues, settings, outputStem, nSubjects);
makeObservedPredictedFigure(ppcValues, settings, outputStem);
makeResidualFigure(ppcValues, settings, outputStem);

disp(metrics)
end

function settings = applyDefaults(settings, rootDir, outcomeType)
if isfield(settings, 'modelStem')
    modelStem = settings.modelStem;
else
    modelStem = 'ConflictHistoryRL';
end
defaults.nReplications = 500;
defaults.randomSeed = 20260731;
defaults.historyLength = 3;
defaults.nSubjectsRequested = Inf;
defaults.excludeCrossBlockHistories = true;
defaults.resampleOutsideRTRange = true;
defaults.minRT = 60;
defaults.maxRT = 1200;
defaults.figureVisible = 'off';
defaults.modelStem = modelStem;
defaults.dataDir = fullfile(rootDir, 'CC_Models', 'data');
defaults.fitDir = fullfile(rootDir, 'CC_RL_Models', ...
    modelStem, outcomeType, 'SubjectData');
defaults.fitPattern = ['*_' modelStem '_' outcomeType '.mat'];
defaults.outputDir = fullfile(rootDir, 'Figures', ...
    modelStem, [outcomeType '_PPC']);
defaults.outputStem = ['PPC_' modelStem '_' outcomeType];
names = fieldnames(defaults);
for nameIndex = 1:numel(names)
    name = names{nameIndex};
    if ~isfield(settings, name), settings.(name) = defaults.(name); end
end
end

function [labels, historyLabels, current, historyCount] = ...
    makeConflictSequenceLabels(historyLength)
nBits = historyLength + 1;
nSequences = 2 ^ nBits;
labels = strings(nSequences, 1);
historyLabels = strings(nSequences, 1);
current = zeros(nSequences, 1);
historyCount = zeros(nSequences, 1);
for sequence = 0:nSequences - 1
    bits = dec2bin(sequence, nBits) - '0';
    letters = repmat('C', 1, nBits);
    letters(bits == 1) = 'I';
    labels(sequence + 1) = strjoin(cellstr(letters'), '-');
    historyLabels(sequence + 1) = char(letters(2:end));
    current(sequence + 1) = bits(1);
    historyCount(sequence + 1) = sum(bits(2:end));
end
end

function [sequenceCode, historyCount] = ...
    encodeConflictHistory(incongruent, currentTrial, historyLength)
sequenceCode = zeros(size(currentTrial));
historyCount = zeros(size(currentTrial));
for lag = 0:historyLength
    value = incongruent(currentTrial - lag);
    sequenceCode = sequenceCode + value * 2 ^ (historyLength - lag);
    if lag > 0, historyCount = historyCount + value; end
end
invalid = ~isfinite(sequenceCode) | ~isfinite(historyCount);
sequenceCode(invalid) = nan;
historyCount(invalid) = nan;
end

function simulated = simulateRT(mu, sigma, nRep, minRT, maxRT, resample)
simulated = exp(mu + sigma * randn(numel(mu), nRep));
if ~resample, return; end
outside = simulated <= minRT | simulated >= maxRT;
muMatrix = repmat(mu, 1, nRep);
attempt = 0;
while any(outside(:)) && attempt < 100
    simulated(outside) = exp(muMatrix(outside) + ...
        sigma * randn(sum(outside(:)), 1));
    outside = simulated <= minRT | simulated >= maxRT;
    attempt = attempt + 1;
end
simulated = min(max(simulated, minRT + eps), maxRT - eps);
end

function divided = divideReplicated(values, counts)
divided = values;
for cellIndex = 1:numel(counts)
    if counts(cellIndex) > 0
        divided(:, cellIndex) = values(:, cellIndex) / counts(cellIndex);
    else
        divided(:, cellIndex) = nan;
    end
end
end

function divided = divideReplicatedCounts(values, counts)
divided = values;
for count = 1:size(counts, 1)
    for condition = 1:size(counts, 2)
        if counts(count, condition) > 0
            divided(:, count, condition) = ...
                values(:, count, condition) / counts(count, condition);
        else
            divided(:, count, condition) = nan;
        end
    end
end
end

function makeHistoryCellFigure(P, settings, stem, nSubjects)
fig = figure('Visible', settings.figureVisible, 'Color', 'w', ...
    'Position', [100 100 1500 650]);
colors = [0.22 0.48 0.75; 0.86 0.38 0.24];
conditionNames = {'Current congruent', 'Current incongruent'};
for condition = 0:1
    subplot(1, 2, condition + 1)
    useCells = find(P.sequenceCurrent == condition);
    observed = P.observedGroupCells(useCells);
    predicted = P.predictedMedianCells(useCells);
    lower = P.predictedCICells(1, useCells);
    upper = P.predictedCICells(2, useCells);
    bar(1:numel(useCells), observed, 0.72, ...
        'FaceColor', colors(condition + 1, :), 'EdgeColor', 'none');
    hold on
    errorbar(1:numel(useCells), predicted, predicted - lower, ...
        upper - predicted, 'ko', 'MarkerFaceColor', 'w', ...
        'LineWidth', 1.2, 'MarkerSize', 5, 'CapSize', 5)
    xticks(1:numel(useCells)); xticklabels(P.historyLabels(useCells));
    xlabel('Congruence history: t-1, t-2, t-3')
    ylabel(outcomeLabel(P.outcomeType))
    title(conditionNames{condition + 1})
    box off; grid on
end
sgtitle(sprintf('Conflict-history PPC (%s, N = %d): bars observed; points model', ...
    P.outcomeType, nSubjects))
savePDF(fig, fullfile(settings.outputDir, [stem '_HistoryCells.pdf']))
close(fig)
end

function makeConflictCountFigure(P, settings, stem, nSubjects)
fig = figure('Visible', settings.figureVisible, 'Color', 'w', ...
    'Position', [100 100 1400 600]);
colors = [0.22 0.48 0.75; 0.86 0.38 0.24];
x = 0:settings.historyLength;
conditionNames = {'Current congruent', 'Current incongruent'};
for condition = 1:2
    subplot(1, 2, condition)
    observed = P.observedGroupCounts(:, condition);
    predicted = P.predictedMedianCounts(:, condition);
    lower = squeeze(P.predictedCICounts(1, :, condition))';
    upper = squeeze(P.predictedCICounts(2, :, condition))';
    plot(x, observed, '-o', 'Color', colors(condition, :), ...
        'MarkerFaceColor', colors(condition, :), 'LineWidth', 2); hold on
    errorbar(x, predicted, predicted - lower, upper - predicted, ...
        '--s', 'Color', [0.15 0.15 0.15], 'MarkerFaceColor', 'w', ...
        'LineWidth', 1.5, 'CapSize', 6)
    xlabel('Number of incongruent trials at t-1:t-3')
    ylabel(outcomeLabel(P.outcomeType))
    title(conditionNames{condition}); xticks(x); box off; grid on
    legend({'Observed', 'Model'}, 'Location', 'best', 'Box', 'off')
end
sgtitle(sprintf('Accumulated conflict-history effect (%s, N = %d)', ...
    P.outcomeType, nSubjects))
savePDF(fig, fullfile(settings.outputDir, [stem '_ConflictCount.pdf']))
close(fig)
end

function makeObservedPredictedFigure(P, settings, stem)
fig = figure('Visible', settings.figureVisible, 'Color', 'w', ...
    'Position', [100 100 1200 560]);
colors = [0.22 0.48 0.75; 0.86 0.38 0.24];
conditionNames = {'Congruent', 'Incongruent'};
for condition = 0:1
    subplot(1, 2, condition + 1)
    useCells = P.sequenceCurrent == condition;
    x = P.observedGroupCells(useCells);
    y = P.predictedMedianCells(useCells);
    scatter(x, y, 55, colors(condition + 1, :), 'filled'); hold on
    limits = [min([x(:); y(:)]), max([x(:); y(:)])];
    padding = max(diff(limits) * 0.08, eps);
    limits = limits + [-padding padding];
    plot(limits, limits, '--k', 'LineWidth', 1.2)
    xlim(limits); ylim(limits); axis square
    xlabel(['Observed ' outcomeLabel(P.outcomeType)])
    ylabel(['Model ' outcomeLabel(P.outcomeType)])
    title(sprintf('%s\nr = %.3f, RMSE = %.3f', ...
        conditionNames{condition + 1}, ...
        P.metrics.Correlation(condition + 1), ...
        P.metrics.RMSE(condition + 1)))
    box off; grid on
end
sgtitle('Observed versus model-predicted conflict-history cells')
savePDF(fig, fullfile(settings.outputDir, [stem '_ObservedVsPredicted.pdf']))
close(fig)
end

function makeResidualFigure(P, settings, stem)
fig = figure('Visible', settings.figureVisible, 'Color', 'w', ...
    'Position', [100 100 1400 600]);
colors = [0.22 0.48 0.75; 0.86 0.38 0.24];
x = 0:settings.historyLength;
conditionNames = {'Current congruent', 'Current incongruent'};
for condition = 1:2
    subplot(1, 2, condition)
    values = P.residualGroupCounts(:, condition);
    plot(x, values, '-o', 'Color', colors(condition, :), ...
        'MarkerFaceColor', colors(condition, :), 'LineWidth', 2); hold on
    yline(0, '--k'); xticks(x)
    xlabel('Number of incongruent trials at t-1:t-3')
    if strcmp(P.outcomeType, 'RT')
        ylabel('Residual log RT')
    else
        ylabel('Residual error (pp)')
    end
    title(conditionNames{condition}); box off; grid on
end
sgtitle('Residual conflict-history effect')
savePDF(fig, fullfile(settings.outputDir, [stem '_ResidualHistory.pdf']))
close(fig)
end

function label = outcomeLabel(outcomeType)
if strcmp(outcomeType, 'RT')
    label = 'RT (ms)';
else
    label = 'Error rate (%)';
end
end

function savePDF(fig, fileName)
set(findall(fig, '-property', 'FontName'), 'FontName', 'Arial')
position = get(fig, 'Position');
paperSize = position(3:4) / 100;
set(fig, 'PaperUnits', 'inches', 'PaperSize', paperSize, ...
    'PaperPosition', [0 0 paperSize], 'PaperPositionMode', 'manual')
print(fig, fileName, '-dpdf', '-painters', '-r300')
end

function values = columnDouble(values, nValues)
values = double(values(1:nValues));
values = values(:);
end
