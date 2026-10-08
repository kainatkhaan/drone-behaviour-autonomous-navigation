import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import os

output_dir = r"c:\Users\BN Com\Downloads\kainat research paper\output"
os.makedirs(output_dir, exist_ok=True)
dataset_path = os.path.join(output_dir, "drone_navigation_dataset.csv")

print("Loading dataset...")
df = pd.read_csv(dataset_path)

print("1. Data Exploration Plots")
# Class Distribution
plt.figure(figsize=(10, 6))
sns.countplot(y='Navigation_Action', data=df, order=df['Navigation_Action'].value_counts().index)
plt.title("Navigation Action Class Distribution")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig_class_distribution.png"), dpi=150, bbox_inches='tight')
plt.close()

# Correlation Matrix
plt.figure(figsize=(16, 12))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig_correlation_matrix.png"), dpi=150, bbox_inches='tight')
plt.close()

# Feature Distributions
features_to_plot = ['LiDAR_Front', 'Velocity', 'Altitude', 'IMU_Roll']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, feature in enumerate(features_to_plot):
    if feature in df.columns:
        sns.histplot(df[feature], kde=True, ax=axes[idx//2, idx%2])
        axes[idx//2, idx%2].set_title(f'Distribution of {feature}')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig_feature_distributions.png"), dpi=150, bbox_inches='tight')
plt.close()

# Pairplot
features_for_pairplot = features_to_plot + ['Navigation_Action']
if all(f in df.columns for f in features_for_pairplot):
    sns.pairplot(df[features_for_pairplot], hue='Navigation_Action')
    plt.savefig(os.path.join(output_dir, "fig_pairplot_features.png"), dpi=150, bbox_inches='tight')
    plt.close()

print("2. Preprocessing")
X = df.drop(columns=['Navigation_Action'])
y = df['Navigation_Action']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
classes = label_encoder.classes_

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
print(f"Dataset shape: {X.shape}")

print("3. Train Models")
models = {
    'SVM': SVC(kernel='rbf'),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

results = []

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Macro Precision': prec,
        'Macro Recall': rec,
        'Macro F1': f1
    })
    
    print(f"Classification Report for {name}:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"fig_cm_{name}.png"), dpi=150, bbox_inches='tight')
    plt.close()

print("4. Comparison Results")
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, "results_comparison.csv"), index=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=results_df)
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig_accuracy_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()

results_melted = pd.melt(results_df, id_vars=['Model'], value_vars=['Macro Precision', 'Macro Recall', 'Macro F1'],
                         var_name='Metric', value_name='Score')

plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='Score', hue='Metric', data=results_melted)
plt.title("Model Performance Metrics Comparison")
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig_f1_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()

print("\n--- Final Results Comparison ---")
print(results_df.to_string())

print("5. Feature Importance")
rf_model = models['RandomForest']
importances_rf = rf_model.feature_importances_
indices_rf = np.argsort(importances_rf)[::-1]

plt.figure(figsize=(12, 8))
sns.barplot(x=importances_rf[indices_rf], y=X.columns[indices_rf])
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig_feature_importance_rf.png"), dpi=150, bbox_inches='tight')
plt.close()

xgb_model = models['XGBoost']
importances_xgb = xgb_model.feature_importances_
indices_xgb = np.argsort(importances_xgb)[::-1]

plt.figure(figsize=(12, 8))
sns.barplot(x=importances_xgb[indices_xgb], y=X.columns[indices_xgb])
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig_feature_importance_xgb.png"), dpi=150, bbox_inches='tight')
plt.close()

print("Experiment 1 complete.")
