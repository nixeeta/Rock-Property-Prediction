# 3_visualizations.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        sorted_idx = importance.argsort()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance[sorted_idx], y=[feature_names[i] for i in sorted_idx])
        plt.title("Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

def plot_accuracy_comparison(results):
    model_names = list(results.keys())
    accuracies = [results[name]["accuracy"] for name in model_names]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=model_names, y=accuracies, palette="Set2")
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
