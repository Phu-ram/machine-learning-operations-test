import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


def evaluate_model(y_test, y_predicted):
    sns.regplot(
        y_test, y_predicted, scatter_kws={"color": "black"}, line_kws={"color": "red"}
    )
    r2 = r2_score(y_test, y_predicted)
    mse = mean_squared_error(y_test, y_predicted)
    print("Test R2: ", r2)
    print("Test MSE: ", mse)


def grid_search(model, grid, X_train, y_train, cv):
    search = GridSearchCV(estimator=model, param_grid=grid, cv=cv, scoring="r2")
    grid_result = search.fit(X_train, y_train)
    # summarize results
    print(
        "Best training R2: %f using %s"
        % (grid_result.best_score_, grid_result.best_params_)
    )
    best_model = grid_result.best_estimator_
    return best_model
