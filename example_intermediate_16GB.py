'''
Unless noted otherwise all code created by the team at Numerai.

'''

import time
start = time.time()

import numpy as np 
import pandas as pd
from xgboost import XGBRegressor
import gc
import json

from numerapi import NumerAPI
from halo import Halo
from utils import (
    save_model, 
    load_model, 
    neutralize, 
    get_biggest_change_features, 
    ERA_COL,
    DATA_TYPE_COL,
    TARGET_COL, # currently only works with nomi_20
    era_boost_train 
    )

napi = NumerAPI()
spinner = Halo(text='', spinner='dots')

current_round = napi.get_current_round(tournament=8)  # tournament 8 is the primary Numerai Tournament

# napi.list_datasets()

# Tournament data changes every week so we specify the round in their name. Training
# and validation data only change periodically, so no need to download them every time.
print('Downloading dataset files...')
napi.download_dataset("numerai_training_data_int8.parquet", "training_data_int8.parquet")
napi.download_dataset("numerai_tournament_data_int8.parquet", f"tournament_data_int8_{current_round}.parquet")
napi.download_dataset("numerai_validation_data_int8.parquet", f"validation_data_int8.parquet")
napi.download_dataset("example_validation_predictions.parquet", "example_validation_predictions.parquet")
napi.download_dataset("features.json", "features.json")

print('Reading xlsmall training data')
# read the feature2 metadata and get the "xlsmall" feature set optimized for 16GB Machines
with open("features2.json", "r") as f: 
    feature_metadata = json.load(f)

# select feature set
features = feature_metadata["feature_sets"]["xlsmall"]

# read in just those features along with era and target columns
read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
training_data = pd.read_parquet('training_data_int8.parquet', columns=read_columns)

# pare down the number of eras to every 4th era
every_4th_era = training_data[ERA_COL].unique()[::4]
training_data = training_data[training_data[ERA_COL].isin(every_4th_era)]

# getting the per era correlation of each feature vs the target
all_feature_corrs = training_data.groupby(ERA_COL).apply(
    lambda era: era[features].corrwith(era[TARGET_COL])
)

features = [c for c in training_data if c.startswith("feature_")]

# select model for prediction
model_name = f"model_target" # the final iteration

# alternatively, call a different iteration after your initial run.
# model_name = f"md6_ne500_ni20_target_nomi_20" 

# load model if present
print(f"loading {model_name}")
model = load_model(model_name)
if not model:
    print(f"model not found, training new one")
    for md in [6]: # max depth
        for ne in [500]: # num estimators
            for lr in [0.001]: # learning rate
                for cs in [0.1]: # colsample by tree
                    for ni in [15]: # number of iterations
                        model = era_boost_train(
                            training_data[features], training_data[TARGET_COL], era_col=training_data["era"], proportion=0.5, md=md, ne=ne, ni=ni, lr=lr, cs=cs)
                        gc.collect()

save_model(model, model_name)

# getting the per era correlation of each feature vs the target
all_feature_corrs = training_data.groupby(ERA_COL).apply(
    lambda era: era[features].corrwith(era[TARGET_COL])
)

# find the riskiest features by comparing their correlation vs
# the target in each half of training data; we'll use these later
riskiest_features = get_biggest_change_features(all_feature_corrs, 50)

# "garbage collection" (gc) gets rid of unused data and frees up memory
gc.collect()

print('Reading minimal features of validation and tournament data...')
validation_data = pd.read_parquet('validation_data_int8.parquet',
                                  columns=read_columns)
tournament_data = pd.read_parquet(f'tournament_data_int8_{current_round}.parquet',
                                  columns=read_columns)
nans_per_col = tournament_data[tournament_data["data_type"] == "live"].isna().sum()

# check for nans and fill nans
if nans_per_col.any():
    total_rows = len(tournament_data[tournament_data["data_type"] == "live"])
    print(f"Number of nans per column this week: {nans_per_col[nans_per_col > 0]}")
    print(f"out of {total_rows} total rows")
    print(f"filling nans with 2")
    tournament_data.loc[:, features].fillna(2, inplace=True)
else:
    print("No nans in the features this week!")

spinner.start('Predicting on validation and tournament data')
# double check the feature that the model expects vs what is available to prevent our
# pipeline from failing if Numerai adds more data and we don't have time to retrain!
model_expected_features = model.get_booster().feature_names
if set(model_expected_features) != set(features):
    print(f"New features are available! Might want to retrain model {model_name}.")
validation_data.loc[:, f"preds_{model_name}"] = model.predict(
    validation_data.loc[:, model_expected_features])
tournament_data.loc[:, f"preds_{model_name}"] = model.predict(
    tournament_data.loc[:, model_expected_features])
spinner.succeed()

gc.collect()

spinner.start('Neutralizing to risky features')

# neutralize our predictions to the riskiest features
validation_data[f"preds_{model_name}_neutral_riskiest_50"] = neutralize(
    df=validation_data,
    columns=[f"preds_{model_name}"],
    neutralizers=riskiest_features,
    proportion=1.0,
    normalize=True,
    era_col=ERA_COL
)

tournament_data[f"preds_{model_name}_neutral_riskiest_50"] = neutralize(
    df=tournament_data,
    columns=[f"preds_{model_name}"],
    neutralizers=riskiest_features,
    proportion=1.0,
    normalize=True,
    era_col=ERA_COL
)
spinner.succeed()

model_to_submit = f"preds_{model_name}_neutral_riskiest_50"

# rename best model to "prediction" and rank from 0 to 1 to meet upload requirements
validation_data["prediction"] = validation_data[model_to_submit].rank(pct=True)
tournament_data["prediction"] = tournament_data[model_to_submit].rank(pct=True)
validation_data["prediction"].to_csv(f"validation_predictions_{current_round}_8.csv")
tournament_data["prediction"].to_csv(f"tournament_predictions_{current_round}.csv")

spinner.start('Reading example validationa predictions')
validation_preds = pd.read_parquet('example_validation_predictions.parquet')
validation_data[EXAMPLE_PREDS_COL] = validation_preds["prediction"]
spinner.succeed()

# get some stats about each of our models to compare...
# fast_mode=True so that we skip some of the stats that are slower to calculate
validation_stats = validation_metrics(validation_data, [model_to_submit], example_col=EXAMPLE_PREDS_COL, fast_mode=True)
print(validation_stats[["mean", "sharpe"]].to_markdown())

print(f'done in {(time.time() - start) / 60} mins')
