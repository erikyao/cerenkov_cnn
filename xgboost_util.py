from xgboost import XGBClassifier

_RANDOM_STATE = 1337

_CERENKOV3_XGBOOST_PARAMS = dict(max_depth=10, learning_rate=0.1, n_estimators=100, gamma=10, subsample=1.0, colsample_bytree=0.3)

def new_XGBoost_classifier(rare_event_rate):
    clf = XGBClassifier(verbosity=0, objective='binary:logistic', booster='gbtree', n_jobs=-1, scale_pos_weight=1, base_score=rare_event_rate, random_state=_RANDOM_STATE)
    clf = clf.set_params(**_CERENKOV3_XGBOOST_PARAMS)
    return clf

def get_XGBoost_proba(clf, x_train, y_train, x_test, y_test=None):
    clf.fit(x_train, y_train)
    
    proba_train = clf.predict_proba(x_train)
    proba_test = clf.predict_proba(x_test)

    pos_index = [i for i, class_ in enumerate(clf.classes_) if class_ == 1][0]

    return proba_train[:, pos_index], proba_test[:, pos_index]
