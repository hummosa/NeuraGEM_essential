# Conflict-prediction RL and target-sequence DBM package

This folder contains both conflict-prediction variants and the selected target-sequence model for reaction time (RT) and accuracy, together with behavioral data, completed group-level fits, model-comparison results, and parametric predictive checks (PPCs).

## Selected models

Both the composite conflict-history RL model and the congruency-only RL model are included for both RT and accuracy. The table below indicates which variant has the lower summed BIC for each outcome; it is not used to omit the alternative model.

| Analysis | Selected model | Reason for inclusion |
|---|---|---|
| Conflict prediction, RT | Composite conflict-history RL | It has the lower summed BIC for RT. The comparison with the congruency-only version is weak at the participant level, so this should not be described as a decisive winner. |
| Conflict prediction, accuracy | Congruency-only RL | It has the lower summed BIC for errors. Again, the participant-level comparison is inconclusive. |
| Target-sequence DBM, RT | Direction-specific DBM | Preferred to the symmetric DBM by BIC for 62.2% of participants; summed delta BIC = -2264.31. |
| Target-sequence DBM, accuracy | Direction-specific DBM | Preferred to the symmetric DBM by BIC for 79.0% of participants; summed delta BIC = -4246.91. |

For the conflict comparison, delta BIC is defined as congruency-only minus composite. The RT summed delta BIC is +235.58, favoring the composite model; the error summed delta BIC is -119.51, favoring the congruency-only model. Median effects and signed-rank tests do not distinguish the two conflict models reliably (RT p = .108; error p = .808).

The accuracy analyses are named `Error` in the code because they model error probability directly: `Error = 1` denotes an incorrect response and `Error = 0` a correct response.

## Folder contents

- `CC_Models/data/`: behavioral data for 998 participants.
- `CC_RL_Models/`: completed group-level fit files for both conflict-model variants on both outcomes, plus the directional DBM on both outcomes (six fits in total).
- `Figures/`: precomputed PPC figures and values, plus the model-comparison outputs used for selection.
- Root-level `.m` files: model-fitting and PPC code.

The PPC abbreviation is used throughout the code. These are parametric predictive checks conditional on fitted point estimates, rather than fully Bayesian posterior predictive checks.

## Running the analyses

Open MATLAB, make this folder the current folder, and run the fitting scripts in this order:

1. `RL_ConflictHistory_RT.m`
2. `RL_ConflictHistory_Error.m`
3. `RL_CongruencyOnly_RT.m`
4. `RL_CongruencyOnly_Error.m`
5. `DBM_TargetSequence_Directional_RT.m`
6. `DBM_TargetSequence_Directional_Error.m`

After the fits have completed, run the corresponding predictive checks:

1. `PPC_ConflictHistoryRL_RT.m`
2. `PPC_ConflictHistoryRL_Error.m`
3. `PPC_CongruencyOnlyRL_RT.m`
4. `PPC_CongruencyOnlyRL_Error.m`
5. `PPC_TargetSequenceDirectionalDBM_RT.m`
6. `PPC_TargetSequenceDirectionalDBM_Error.m`

The completed group result files and all precomputed PPC outputs are already included. Participant-level fit files (`SubjectData`) are omitted to keep the package reasonably small. The fitting scripts regenerate them; they must exist before recomputing the PPCs.

The fitting scripts resume from participant-level results when available. On a fresh copy of this package, they therefore refit all participants and create the required output folders automatically.

## Model summaries

### Conflict-prediction RL

The latent expected-conflict state is updated trial by trial using a delta rule. The RT observation model predicts log RT on correct trials. The accuracy model predicts error probability with logistic regression. Both include current incongruence, its interaction with expected conflict, and nuisance regressors.

The composite model updates expected conflict from congruency and stimulus distance. The congruency-only model updates it from congruency alone, while retaining distance as a nuisance regressor.

### Direction-specific target-sequence DBM

The model maintains separate beliefs about the probability of a right target following a left versus a right target. A shared persistence parameter controls both beliefs. Trialwise unexpectedness, repetition/alternation, congruence, their interaction, and nuisance regressors predict RT or error probability.
