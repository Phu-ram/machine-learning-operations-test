import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")


def merge_data(
    orders_filename: str, restuarants_filename: str, foreign_key: str = "restaurant_id"
) -> pd.DataFrame:
    orders = pd.read_csv(orders_filename)
    restaurants = pd.read_csv(restuarants_filename)
    return orders.merge(restaurants, how="left", on=foreign_key)


def extract_datetime():
    NotImplemented


def remove_outliers_iqr(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    print("{column_name} Q1: {Q1}".format(column_name=column_name, Q1=Q1))
    print("{column_name} Q3: {Q3}".format(column_name=column_name, Q3=Q3))
    print("{column_name} IQR: {IQR}".format(column_name=column_name, IQR=IQR))
    # Dropping the Outliers
    df = df[
        ~((df[column_name] < (Q1 - 1.5 * IQR)) | (df[column_name] > (Q3 + 1.5 * IQR)))
    ]
    return df
