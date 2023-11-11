import warnings

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score, train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


def load_dataset(orders_filename: str, restaurants_filename: str, **kwargs):
    """
    :param orders_filename: file_path of the order file
    :param restaurants_filename: file_path of the restaurants file
    :param kwargs: pandas kwargs
    :return: pandas dataFrame
    """
    df_orders = pd.read_csv(orders_filename, **kwargs)
    df_restaurants = pd.read_csv(restaurants_filename, **kwargs)
    df = df_orders.merge(df_restaurants, how="left", on="restaurant_id")
    return df


def check_missing_value(df):
    return df.isnull().sum(axis=0)


def encode(data: pd.DataFrame, col: str, max_val: int) -> pd.DataFrame:
    """Cyclical features encoding - hour, monthday, weekday
    :param data: Input dataframe
    :param col: column name
    :param max_val: max value of the encoded column
    :return:
    """
    data[col + "_s"] = np.sin(2 * np.pi * data[col] / max_val)
    data[col + "_c"] = np.cos(2 * np.pi * data[col] / max_val)
    return data


def remove_outliers_iqr(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Detect IQR and remove outliers

    :param df: Input dataframe
    :param column_name: The column that outlier will be removed
    :return: pandas DataFrame
    """
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    print("{column_name} Q1: {Q1}".format(column_name=column_name, Q1=Q1))
    print("{column_name} Q3: {Q3}".format(column_name=column_name, Q3=Q3))
    print("{column_name} IQR: {IQR}".format(column_name=column_name, IQR=IQR))
    # Dropping the Outliers
    df = df[~((df[column_name] < (Q1 - 1.5 * IQR)) | (df[column_name] > (Q3 + 1.5 * IQR)))]
    return df


# evaluate a give model using cross-validation
def evaluate_model_s(model, X_train, y_train):
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X_train, y_train, scoring="r2", cv=3)
    return scores


# get a stacking ensemble of models
def get_stacking_models(gbr, rf, cat, xgb, final):
    # define the base models
    level0 = list()
    level0.append(("gbr", gbr))
    level0.append(("rf", rf))
    level0.append(("cat", cat))
    level0.append(("xgb", xgb))
    # define meta learner model
    level1 = final
    # define the stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=3)
    return model


# get a list of models to evaluate
def get_models():
    rf = RandomForestRegressor(n_estimators=100, min_samples_split=6, max_features="log2")
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, min_samples_split=2)
    cat = CatBoostRegressor(depth=6, iterations=100, learning_rate=0.1, silent=True)
    xgb = XGBRegressor(max_depth=4, n_estimators=100, learning_rate=0.1)
    stacking = get_stacking_models(gbr, rf, cat, xgb, Ridge(alpha=0.3))
    models = dict()
    models["rf"] = rf
    models["gbr"] = gbr
    models["cat"] = cat
    models["xgb"] = xgb
    models["stacking"] = stacking
    return models


def evaluate_model(y_test, y_predicted):
    """Evaluate model on test set.

    :param y_test:
    :param y_predicted:
    :return:
    """
    sns.regplot(y_test, y_predicted, scatter_kws={"color": "black"}, line_kws={"color": "red"})
    r2 = r2_score(y_test, y_predicted)
    mse = mean_squared_error(y_test, y_predicted)
    print("Test R2: ", r2)
    print("Test MSE: ", mse)


def grid_search(model, grid, X_train, y_train, cv):
    """Define grid search
    :param model:
    :param grid:
    :param X_train:
    :param y_train:
    :param cv:
    :return:
    """
    search = GridSearchCV(estimator=model, param_grid=grid, cv=cv, scoring="r2")
    grid_result = search.fit(X_train, y_train)
    # summarize results
    print("Best training R2: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    best_model = grid_result.best_estimator_
    return best_model


def transform_dataset(df, restaurant_mean):
    df = df.merge(restaurant_mean, how="left", on="restaurant_id", suffixes=("", "1"))
    df.dropna(inplace=True)
    columns = [
        "monthday",
        "hour",
        "weekday",
        "city_id",
        "country_id",
        "type_of_food_id",
        "restaurant_id",
        "r_counts",
        "order_value_gbp",
        "number_of_items",
        "prep_time_seconds1",  # restaurant avg prep time
        "prep_time_seconds",
    ]
    df = df[columns]
    y = df["prep_time_seconds"]
    X = df.drop(columns=["prep_time_seconds"])
    return X, y
