from src.data_preprocessing import load_iris, split_data, scale_data
from src.train_models import train_logistic_regression, train_decision_tree, gridsearch_decision_tree
from src.evaluation import evaluate_model
from src.eda import plot_pairplot, check_missing_and_duplicates

def main():

    X, y, iris = load_iris()

    df = X.copy()
    df["species"] = y
    df["species_name"] = df["species"].map(dict(enumerate(iris.target_names)))

    check_missing_and_duplicates(df)
    plot_pairplot(df)

    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    lr = train_logistic_regression(X_train_scaled, y_train)
    print("\n--- Logistic Regression ---")
    evaluate_model(lr, X_test_scaled, y_test, iris.target_names)

    dt = train_decision_tree(X_train, y_train)
    print("\n--- Decision Tree ---")
    evaluate_model(dt, X_test, y_test, iris.target_names)

    best_dt, params = gridsearch_decision_tree(X_train, y_train)
    print("\nMejor modelo con GridSearch:", params)
    evaluate_model(best_dt, X_test, y_test, iris.target_names)

if __name__ == "__main__":
    main()
