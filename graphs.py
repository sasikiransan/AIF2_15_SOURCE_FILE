import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

epochs = np.arange(1, 21)
model_accuracy = np.linspace(0.65, 0.82, 20)
best_model_accuracy = np.linspace(0.70, 0.90, 20)
model_loss = np.linspace(1.2, 0.4, 20)
best_model_loss = np.linspace(1.0, 0.2, 20)

# 1. Accuracy over epochs
plt.figure(figsize=(8, 5))
plt.plot(epochs, model_accuracy, label='Your Model (82%)', marker='o')
plt.plot(epochs, best_model_accuracy, label='Best Model (90%)', linestyle='dashed')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy Over Epochs')
plt.show()

# 2. Loss over epochs
plt.figure(figsize=(8, 5))
plt.plot(epochs, model_loss, label='Your Model', marker='o')
plt.plot(epochs, best_model_loss, label='Best Model', linestyle='dashed')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss Over Epochs')
plt.show()

# 3. Confusion Matrix
y_true = np.random.randint(0, 2, 100)
y_pred = np.random.randint(0, 2, 100)
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 4. Feature Importance Heatmap
features = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']
importance = np.random.rand(5)
sns.barplot(x=importance, y=features)
plt.xlabel('Importance Score')
plt.title('Feature Importance')
plt.show()

# 5. ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='dashed')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# 6. Bar chart for model performance comparison
models = ['Your Model', 'Best Model']
performance = [82, 90]
sns.barplot(x=models, y=performance)
plt.ylabel('Accuracy (%)')
plt.title('Model Performance Comparison')
plt.show()

# 7. Boxplot for Error Distribution
errors = np.random.normal(0, 1, 100)
sns.boxplot(errors)
plt.title('Error Distribution')
plt.show()

# 8. Correlation Heatmap
data = np.random.rand(10, 10)
sns.heatmap(np.corrcoef(data), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# 9. Precision, Recall, and F1-score Comparison
metrics = ['Precision', 'Recall', 'F1-score']
your_model_scores = [0.78, 0.81, 0.80]
best_model_scores = [0.85, 0.88, 0.86]
plt.figure(figsize=(8, 5))
plt.bar(metrics, your_model_scores, label='Your Model', alpha=0.7)
plt.bar(metrics, best_model_scores, label='Best Model', alpha=0.5)
plt.ylabel('Score')
plt.legend()
plt.title('Precision, Recall, and F1-score Comparison')
plt.show()

# 10. Distribution of Predictions vs Actual Values
y_actual = np.random.normal(50, 10, 100)
y_predicted = y_actual + np.random.normal(0, 5, 100)
sns.kdeplot(y_actual, label='Actual', fill=True, alpha=0.5)
sns.kdeplot(y_predicted, label='Predicted', fill=True, alpha=0.5)
plt.xlabel('Value')
plt.title('Actual vs Predicted Distribution')
plt.legend()
plt.show()
