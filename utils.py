import os
import numpy as np
import pandas as pd
import scipy
from halo import Halo
from pathlib import Path
import json
from scipy.stats import skew, kurtosis
from xgboost import XGBRegressor
import ast

ERA_COL = "era"
TARGET_COL = "target_nomi_20"
DATA_TYPE_COL = "data_type"
EXAMPLE_PREDS_COL = "example_preds"

spinner = Halo(text='', spinner='dots')

MODEL_FOLDER = "models"
MODEL_CONFIGS_FOLDER = "model_configs"
PREDICTION_FILES_FOLDER = "prediction_files"


def save_model(model, name):
    try:
        Path(MODEL_FOLDER).mkdir(exist_ok=True, parents=True)
    except Exception as ex:
        pass
    pd.to_pickle(model, f"{MODEL_FOLDER}/{name}.pkl")


def load_model(name):
    path = Path(f"{MODEL_FOLDER}/{name}.pkl")
    if path.is_file():
        model = pd.read_pickle(f"{MODEL_FOLDER}/{name}.pkl")
    else:
        model = False
    return model


def save_model_config(model_config, model_name):
    try:
        Path(MODEL_CONFIGS_FOLDER).mkdir(exist_ok=True, parents=True)
    except Exception as ex:
        pass
    with open(f"{MODEL_CONFIGS_FOLDER}/{model_name}.json", 'w') as fp:
        json.dump(model_config, fp)


def load_model_config(model_name):
    path_str = f"{MODEL_CONFIGS_FOLDER}/{model_name}.json"
    path = Path(path_str)
    if path.is_file():
        with open(path_str, 'r') as fp:
            model_config = json.load(fp)
    else:
        model_config = False
    return model_config


def get_biggest_change_features(corrs, n):
    all_eras = corrs.index.sort_values()
    h1_eras = all_eras[:len(all_eras) // 2]
    h2_eras = all_eras[len(all_eras) // 2:]

    h1_corr_means = corrs.loc[h1_eras, :].mean()
    h2_corr_means = corrs.loc[h2_eras, :].mean()

    corr_diffs = h2_corr_means - h1_corr_means
    worst_n = corr_diffs.abs().sort_values(ascending=False).head(n).index.tolist()
    return worst_n


def get_time_series_cross_val_splits(data, cv = 3, embargo = 12):
    all_train_eras = data[ERA_COL].unique()
    len_split = len(all_train_eras) // cv
    test_splits = [all_train_eras[i * len_split:(i + 1) * len_split] for i in range(cv)]
    # fix the last test split to have all the last eras, in case the number of eras wasn't divisible by cv
    test_splits[-1] = np.append(test_splits[-1], all_train_eras[-1])

    train_splits = []
    for test_split in test_splits:
        test_split_max = int(np.max(test_split))
        test_split_min = int(np.min(test_split))
        # get all of the eras that aren't in the test split
        train_split_not_embargoed = [e for e in all_train_eras if not (test_split_min <= int(e) <= test_split_max)]
        # embargo the train split so we have no leakage.
        # one era is length 5, so we need to embargo by target_length/5 eras.
        # To be consistent for all targets, let's embargo everything by 60/5 == 12 eras.
        train_split = [e for e in train_split_not_embargoed if
                       abs(int(e) - test_split_max) > embargo and abs(int(e) - test_split_min) > embargo]
        train_splits.append(train_split)

    # convenient way to iterate over train and test splits
    train_test_zip = zip(train_splits, test_splits)
    return train_test_zip


def neutralize(df,
               columns,
               neutralizers=None,
               proportion=1.0,
               normalize=True,
               era_col="era"):
    if neutralizers is None:
        neutralizers = []
    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
        df_era = df[df[era_col] == u]
        scores = df_era[columns].values
        if normalize:
            scores2 = []
            for x in scores.T:
                x = (scipy.stats.rankdata(x, method='ordinal') - .5) / len(x)
                x = scipy.stats.norm.ppf(x)
                scores2.append(x)
            scores = np.array(scores2).T
        exposures = df_era[neutralizers].values

        scores -= proportion * exposures.dot(
            np.linalg.pinv(exposures.astype(np.float32)).dot(scores.astype(np.float32)))

        scores /= scores.std(ddof=0)

        computed.append(scores)

    return pd.DataFrame(np.concatenate(computed),
                        columns=columns,
                        index=df.index)


def neutralize_series(series, by, proportion=1.0):
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)

    # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
    exposures = np.hstack(
        (exposures,
         np.array([np.mean(series)] * len(exposures)).reshape(-1, 1)))

    correction = proportion * (exposures.dot(
        np.linalg.lstsq(exposures, scores, rcond=None)[0]))
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized


def unif(df):
    x = (df.rank(method="first") - 0.5) / len(df)
    return pd.Series(x, index=df.index)


def get_feature_neutral_mean(df, prediction_col):
    feature_cols = [c for c in df.columns if c.startswith("feature")]
    df.loc[:, "neutral_sub"] = neutralize(df, [prediction_col],
                                          feature_cols)[prediction_col]
    scores = df.groupby("era").apply(
        lambda x: (unif(x["neutral_sub"]).corr(x[TARGET_COL]))).mean()
    return np.mean(scores)


def fast_score_by_date(df, columns, target, tb=None, era_col="era"):
    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
        df_era = df[df[era_col] == u]
        era_pred = np.float64(df_era[columns].values.T)
        era_target = np.float64(df_era[target].values.T)

        if tb is None:
            ccs = np.corrcoef(era_target, era_pred)[0, 1:]
        else:
            tbidx = np.argsort(era_pred, axis=1)
            tbidx = np.concatenate([tbidx[:, :tb], tbidx[:, -tb:]], axis=1)
            ccs = [np.corrcoef(era_target[tmpidx], tmppred[tmpidx])[0, 1] for tmpidx, tmppred in zip(tbidx, era_pred)]
            ccs = np.array(ccs)

        computed.append(ccs)

    return pd.DataFrame(np.array(computed), columns=columns, index=df[era_col].unique())


def validation_metrics(validation_data, pred_cols, example_col, fast_mode=False):
    validation_stats = pd.DataFrame()
    feature_cols = [c for c in validation_data if c.startswith("feature_")]
    for pred_col in pred_cols:
        # Check the per-era correlations on the validation set (out of sample)
        validation_correlations = validation_data.groupby(ERA_COL).apply(
            lambda d: unif(d[pred_col]).corr(d[TARGET_COL]))

        mean = validation_correlations.mean()
        std = validation_correlations.std(ddof=0)
        sharpe = mean / std

        validation_stats.loc["mean", pred_col] = mean
        validation_stats.loc["std", pred_col] = std
        validation_stats.loc["sharpe", pred_col] = sharpe

        rolling_max = (validation_correlations + 1).cumprod().rolling(window=9000,  # arbitrarily large
                                                                      min_periods=1).max()
        daily_value = (validation_correlations + 1).cumprod()
        max_drawdown = -((rolling_max - daily_value) / rolling_max).max()
        validation_stats.loc["max_drawdown", pred_col] = max_drawdown

        payout_scores = validation_correlations.clip(-0.25, 0.25)
        payout_daily_value = (payout_scores + 1).cumprod()

        apy = (
            (
                (payout_daily_value.dropna().iloc[-1])
                ** (1 / len(payout_scores))
            )
            ** 49  # 52 weeks of compounding minus 3 for stake compounding lag
            - 1
        ) * 100

        validation_stats.loc["apy", pred_col] = apy

        if not fast_mode:
            # Check the feature exposure of your validation predictions
            max_per_era = validation_data.groupby(ERA_COL).apply(
                lambda d: d[feature_cols].corrwith(d[pred_col]).abs().max())
            max_feature_exposure = max_per_era.mean()
            validation_stats.loc["max_feature_exposure", pred_col] = max_feature_exposure

            # Check feature neutral mean
            feature_neutral_mean = get_feature_neutral_mean(validation_data, pred_col)
            validation_stats.loc["feature_neutral_mean", pred_col] = feature_neutral_mean

            # Check top and bottom 200 metrics (TB200)
            tb200_validation_correlations = fast_score_by_date(
                validation_data,
                [pred_col],
                TARGET_COL,
                tb=200,
                era_col=ERA_COL
            )

            tb200_mean = tb200_validation_correlations.mean()[pred_col]
            tb200_std = tb200_validation_correlations.std(ddof=0)[pred_col]
            tb200_sharpe = tb200_mean / tb200_std

            validation_stats.loc["tb200_mean", pred_col] = tb200_mean
            validation_stats.loc["tb200_std", pred_col] = tb200_std
            validation_stats.loc["tb200_sharpe", pred_col] = tb200_sharpe

        # MMC over validation
        mmc_scores = []
        corr_scores = []
        for _, x in validation_data.groupby(ERA_COL):
            series = neutralize_series(unif(x[pred_col]), (x[example_col]))
            mmc_scores.append(np.cov(series, x[TARGET_COL])[0, 1] / (0.29 ** 2))
            corr_scores.append(unif(x[pred_col]).corr(x[TARGET_COL]))

        val_mmc_mean = np.mean(mmc_scores)
        val_mmc_std = np.std(mmc_scores)
        corr_plus_mmcs = [c + m for c, m in zip(corr_scores, mmc_scores)]
        corr_plus_mmc_sharpe = np.mean(corr_plus_mmcs) / np.std(corr_plus_mmcs)

        validation_stats.loc["mmc_mean", pred_col] = val_mmc_mean
        validation_stats.loc["corr_plus_mmc_sharpe", pred_col] = corr_plus_mmc_sharpe

        # Check correlation with example predictions
        per_era_corrs = validation_data.groupby(ERA_COL).apply(lambda d: unif(d[pred_col]).corr(unif(d[example_col])))
        corr_with_example_preds = per_era_corrs.mean()
        validation_stats.loc["corr_with_example_preds", pred_col] = corr_with_example_preds

    # .transpose so that stats are columns and the model_name is the row
    return validation_stats.transpose()


def download_data(napi, filename, dest_path):
    spinner.start(f'Downloading {dest_path}')
    napi.download_dataset(filename, dest_path)
    spinner.succeed()

def ar1(x):
    return np.corrcoef(x[:-1], x[1:])[0,1]

def autocorr_penalty(x):
    n = len(x)
    p = ar1(x)
    return np.sqrt(1 + 2*np.sum ([((n - i)/n)*p**i for i in range(1,n)]))

def smart_sharpe(x):
    return np.mean(x)/(np.std(x, ddof=1)*autocorr_penalty(x))

def spearmanr(target, pred):
    return np.corrcoef(
        target,
        pred.rank(pct=True, method="first")
    )[0, 1]

def era_boost_train(X, y, era_col, proportion, md, lr, cs, ne, ni):
    model = XGBRegressor(max_depth=md, learning_rate=lr, n_estimators=ne, n_jobs=-1, colsample_bytree=cs)
    features = X.columns
    model.fit(X, y)
    new_df = X.copy()
    new_df[TARGET_COL] = y
    new_df["era"] = era_col

    for i in range(ni-1):
        preds = model.predict(X)
        new_df["pred"] = preds
        era_scores = pd.Series(dtype='float32', index=new_df["era"].unique())

        for era in new_df["era"].unique():
            era_df = new_df[new_df["era"] == era]
            era_scores[era] = spearmanr(era_df["pred"], era_df[TARGET_COL])
        
        era_scores.sort_values(inplace=False)
        worst_eras = era_scores[era_scores <= era_scores.quantile(proportion)].index

        worst_df = new_df[new_df["era"].isin(worst_eras)]
        era_scores.sort_index(inplace=True)

        print(f"md{md}_ne{ne}_ni{i}_{TARGET_COL}, auto corr: {ar1(era_scores)}, mean corr: {np.mean(era_scores)}, sharpe: {np.mean(era_scores)/np.std(era_scores)}, smart sharpe: {smart_sharpe(era_scores)}")

        model.n_estimators += ne
        booster = model.get_booster()
        model.fit(worst_df[features], worst_df[TARGET_COL], xgb_model=booster)
        save_model(model, "md" + str(md) + "_ne" + str(ne) + "_ni" + str(i) + "_" + str(TARGET_COL)) # save each iteration as a model for later comparison
        
    return model


def create_features_dict():
    """
    This function generates a dict with all the features from borutashap and saves it as json file for future usage.
    
    Returns:
        dict : the dict with all the features

    The json/dict structure is:
    root:
        - file name: 
                    - feature type (['attributes confirmed important', 'attributes confirmed unimportant', 'temtative attributes remains'])
    
    You can then easily select the features filtering per target type (20, 60, all), feature importance, feature frequency and so on.
    
    Example usage:
    
    import json
    from collections import Counter
    
    with open('features_dict.json') as f:
        features_dict = json.load(f)
    
    important_features = []
    tentative_features = []
    unimportant_features = []

    for key, value in features_dict.items():
        important_features+= value['attributes confirmed important']
        tentative_features += value['tentative attributes remains']
        unimportant_features+= value['attributes confirmed unimportant']

    print(f"Unique important features {len(set(important_features))} out of {len(important_features)}")
    print(f"Unique tentative features {len(set(tentative_features))} out of {len(tentative_features)}")
    print(f"Unique unimportant features {len(set(unimportant_features))} out of {len(unimportant_features)}")
    print(f"Unique important features + tentative features {len(set(important_features+tentative_features))} out of {len(important_features+tentative_features)}")

    important_feat_count = Counter(important_features).most_common()
    tentative_feat_count = Counter(tentative_features).most_common()
    unimportant_feat_count = Counter(unimportant_features).most_common()

    Output:
    Unique important features 345 out of 3517
    Unique possible features 612 out of 4257
    Unique unimportant features 1050 out of 76226
    Unique important features + tentative features 622 out of 7774

    Then we can select the features based on the desired criteria
    cutoff = 5
    selected_features_c5 = list(set([feat for feat, val in important_feat_count if val >= cutoff]+[feat for feat, val in possible_feat_count if val >= cutoff+5]))
    print(f'Selected features that appear at least {cutoff} times as important and {cutoff + 5} as tentative: {len(selected_features_c5)}')

    Output:
    Selected features that appear at least 5 times as important and 10 as tentative: 258

    # Or
    cutoff = 80 # The files are 80 in total (20 targets x 4 eras) so these are the features marked as unimportant in each file
    selected_features_c80 = list(set([feat for feat, val in unimportant_feat_count if val >= cutoff]))
    print(f'Selected features that appear at least {cutoff} times as unimportant: {len(selected_features_c80)}')

    Output:
    Selected features that appear at least 80 times as unimportant: 428
    """    
    
    features_dict = {}
    dir_path = "./borutashap/raw_results/"

    for filename in os.listdir(dir_path):
        if filename.endswith(".txt"): 
            group = filename.replace('.txt', '')
            features_dict[group] = {}
            with open(dir_path + filename, 'r') as f:
                for line in f:
                    l1 = line.split(":")
                    # Remove numbers and spaces 
                    feat_group = "".join([i for i in l1[0] if not i.isdigit()]).strip()
                    # Safe transform string into list
                    feat_list = ast.literal_eval(l1[1].strip())
                    features_dict[group][feat_group] = feat_list
                
    with open('features_dict.json', 'w') as f:
        json.dump(features_dict, f)

    return features_dict