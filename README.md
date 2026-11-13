# Person-Level Fatality Prediction in Traffic Motor Vehicle Crashes: A Machine Learning Approach Using FARS 2023 Data

## The question I started with

After a crash, can the facts recorded at the scene help predict whether a person involved will die? That is the question I set out to answer with FARS 2023, the Fatality Analysis Reporting System from the National Highway Traffic Safety Administration (NHTSA). I wanted a careful, honest answer, not a flashy one, so I tested three models the same way, across ten different random seeds, and I reported the full spread of results instead of only the best run.

## Why it matters

Every year, tens of thousands of people die on US roads. If the patterns in crash records can separate higher risk cases from lower risk ones, that knowledge can help emergency responders, road safety researchers, and policy makers ask better questions about who is most at risk and why. A model will never replace a paramedic or a doctor, and it will never undo a crash. But a clear, reproducible test of what crash data can and cannot predict is a useful starting point, and that is what this project is.

## What I did

I worked in Python, in Google Colab, with the FARS 2023 National files from NHTSA: `accident.csv`, `person.csv`, `vehicle.csv`, and `weather.csv`, included in this repo as `FARS2023NationalCSV.zip`.

1. **I loaded the records.** The files contained 37,654 crashes, 92,400 persons, and 58,319 vehicles.
2. **I defined the target.** For each person, did they die as a result of the crash? I excluded people recorded as having died prior to the crash, because they were not alive at the time of the crash and cannot tell us about fatality resulting from the crash itself.
3. **I focused on motor-vehicle occupants**, so I could analyze seating and restraint features properly.
4. **I built the features.** I converted the raw fields into 30 binary features, covering the person (age group and sex), seating position, restraint use and airbag deployment, vehicle type and damage, rollover and fire, crash characteristics, road and environment (rural, interstate, curve, surface, lighting, weather, work zone), and driver history (license status, past crashes, DWI and speeding convictions).
5. **I prepared the modelling data.** That gave me 83,113 samples with 30 features. I removed only 2 samples with a missing target value. I kept samples with missing feature values, because XGBoost and LightGBM handle them natively.
6. **I split the data.** I used an 80/20 stratified train-test split, 66,488 samples for training and 16,623 for testing, so the share of deaths stayed the same in both parts: 38.7 percent died, 25,714 in training and 6,429 in testing.
7. **I tuned each model with Optuna**, 10 trials per model, using 5-fold stratified cross-validation. The models were Random Forest, LightGBM, and XGBoost.
8. **I compared the three models across 10 random seeds**, generated from the input string "PLANETEARTH", and I tested the differences with paired t-tests. Reporting ten seeds instead of one is the point. A single lucky split can flatter a model, and I did not want that.

The full work is in two notebooks: `FARS2023_FinalModels_699_JWAQ.ipynb`, the single-seed final models, with rendered results in `FARS2023_FinalModels_699_JA/FARS2023_FinalModels_699_JWAQ.md` and `.html`, and `FARS2023_10Seed_Comparison.ipynb`, the ten-seed comparison, with rendered results in `FARS2023_10Seed_Comparison/FARS2023_10Seed_Comparison.md` and `.html`.

## What I found

The headline metric is ROC-AUC, which measures how well a model ranks a person who died above a person who survived. A score of 0.5 is no better than chance, and 1.0 is perfect. Mean scores across 10 seeds, with standard deviation:

| Model | Mean ROC-AUC | Best single seed | Mean F1-Score | Mean accuracy | Mean precision | Mean recall |
|---|---|---|---|---|---|---|
| XGBoost | 0.8718 ± 0.0021 | 0.8751, seed 9 | 0.7373 | 0.7846 | 0.6956 | 0.7865 |
| LightGBM | 0.8714 ± 0.0023 | 0.8745, seed 9 | 0.7372 | 0.7840 | 0.6942 | 0.7868 |
| Random Forest | 0.8695 ± 0.0020 | 0.8717, seed 9 | 0.7359 | 0.7851 | 0.7018 | 0.7724 |

Three things stand out to me. First, all three models performed closely, within about 0.002 of each other on mean ROC-AUC, so the choice of model matters less than the quality of the data and the evaluation. Second, XGBoost had the highest mean ROC-AUC, 0.8718, and the best single-seed result, 0.8751, which is why I recommend it as the primary model, with LightGBM as a close backup. Third, the results were stable: every model had a standard deviation below 0.003 on ROC-AUC across the ten seeds, and Random Forest had the lowest variance, 0.0020, so it was the most stable, even though its mean was slightly lower.

The paired t-tests support that reading. On ROC-AUC, Random Forest was significantly below both LightGBM and XGBoost, p less than 0.001 in each case, while LightGBM versus XGBoost was not significant, p equals 0.0879. On F1-Score, Random Forest versus XGBoost was significant at p equals 0.0277, and the other two pairs were not. In plain words, XGBoost and LightGBM are effectively tied at the top, and Random Forest trails them by a small but real margin.

I also checked generalization directly. For XGBoost, train ROC-AUC was 0.8799, validation 0.8708, and test 0.8736, a train-test gap of 0.0063, which the notebook records as excellent generalization. For Random Forest, train was 0.8809, validation 0.8677, and test 0.8702, a gap of 0.0107, recorded as acceptable generalization.

## What it means, and who it helps

For emergency response planning and road safety research, a model that ranks risk at about 0.87 ROC-AUC on held-out data is genuinely useful as a screening and research tool, and the precision-recall balance matters here too: the models catch roughly 79 percent of deaths (recall about 0.79) while about 70 percent of their death predictions are correct (precision about 0.70). That trade-off is the honest shape of the problem. No model in this project is close to perfect, and none should be treated as a decision maker at the roadside. What this project gives a reader is a reproducible baseline: three standard models, tuned the same way, compared over ten seeds, with the full numbers on the table.

## How to run it yourself

You need Python, plus the packages installed at the top of the notebooks: scikit-learn, pandas, numpy, matplotlib, seaborn, lightgbm, xgboost, optuna, and joblib.

```bash
pip install -q scikit-learn pandas numpy matplotlib seaborn lightgbm xgboost optuna joblib
```

Then open `FARS2023_FinalModels_699_JWAQ.ipynb` to run the single-seed final models, and `FARS2023_10Seed_Comparison.ipynb` to reproduce the ten-seed comparison. The data comes from NHTSA, https://www.nhtsa.gov/file-downloads?p=nhtsa/downloads/FARS/2023/National/, and a copy is in this repo as `FARS2023NationalCSV.zip`. The rendered `.md` and `.html` outputs in `FARS2023_FinalModels_699_JA/` and `FARS2023_10Seed_Comparison/` let you read the full results without running anything.

## What the limits are, stated honestly

- This is a retrospective prediction on 2023 US crash records only. It does not show that any feature causes a death, and it does not generalize to other years or countries without retesting.
- The outcome is death versus survival among motor-vehicle occupants in the FARS records. It is not a triage tool, and it is not ready for clinical or emergency deployment.
- People who died prior to the crash were excluded, and the study focuses on occupants, so pedestrians and other road users outside that group are not covered by these numbers.
- Features were converted to binary form, which simplifies the raw FARS fields. That makes the models easier to compare, and it also throws away detail.
- 98,403 feature values were missing across the clean dataset, and they were kept in place for the models to handle natively. A different missing-data treatment could change the numbers.
- Hyperparameter tuning used 10 Optuna trials per model. More trials might find slightly better settings, and the ten-seed comparison, not the single best run, is the result to trust.
- The models are close together, and the gap between XGBoost and LightGBM is not statistically significant. Picking a winner between those two on this data alone would overstate the evidence.

## Data source

FARS 2023, Fatality Analysis Reporting System, National Highway Traffic Safety Administration (NHTSA). Dataset obtained from https://www.nhtsa.gov/file-downloads?p=nhtsa/downloads/FARS/2023/National/
