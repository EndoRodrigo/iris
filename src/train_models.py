from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def gridsearch_decision_tree(X_train, y_train):
    params = {
        "max_depth": [2, 3, 4, 5, None],
        "min_samples_split": [2, 3, 4, 5],
    }
    gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=5)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_
