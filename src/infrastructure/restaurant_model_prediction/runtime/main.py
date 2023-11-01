from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from numpy import mean, std


# evaluate a give model using cross-validation
def evaluate_model_s(model, X_train, y_train):
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X_train, y_train, scoring='r2', cv=3)
    return scores

# get a stacking ensemble of models
def get_stacking_models(gbr, rf, cat, xgb, final):
    # define the base models
    level0 = list()
    level0.append(('gbr', gbr))
    level0.append(('rf', rf))
    level0.append(('cat', cat))
    level0.append(('xgb', xgb))
    # define meta learner model
    level1 = final
    # define the stacking ensemble
    model = StackingRegressor(estimators = level0, final_estimator = level1, cv = 3)
    return model

# get a list of models to evaluate
def get_models():
    rf = RandomForestRegressor(n_estimators =100, min_samples_split = 6, max_features = 'log2')
    gbr = GradientBoostingRegressor(n_estimators = 100, learning_rate = 0.1, min_samples_split = 2)
    cat = CatBoostRegressor(depth = 6, iterations = 100, learning_rate = 0.1, silent = True)
    xgb = XGBRegressor(max_depth = 4, n_estimators = 100, learning_rate = 0.1)
    stacking = get_stacking_models(gbr, rf, cat, xgb, Ridge(alpha = 0.3))
    models = dict()
    models['rf'] = rf
    models['gbr'] = gbr
    models['cat'] = cat
    models['xgb'] = xgb
    models['stacking'] = stacking
    return models