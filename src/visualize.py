import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def plot_confusion_matrix(cm, labels, title="Confusion Matrix"):
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def plot_decision_tree(model, feature_names, class_names):
    plt.figure(figsize=(12,6))
    plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True)
    plt.show()
