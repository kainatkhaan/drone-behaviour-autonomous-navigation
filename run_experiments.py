import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        pass

plt.rcParams.update({'font.size': 12})
try:
    plt.rcParams.update({'font.family': 'Times New Roman'})
except:
    pass

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.environ.get("OUTPUT_DIR", os.path.join(base_dir, "output"))
os.makedirs(output_dir, exist_ok=True)

# 1. Load Data
dataset_path = os.environ.get("DATASET_PATH", os.path.join(output_dir, 'drone_navigation_dataset.csv'))
if not os.path.exists(dataset_path):
    alt_path = os.path.join(base_dir, 'drone behaviour for autonomous navigation', 'uav_navigation_dataset.csv')
    if os.path.exists(alt_path):
        dataset_path = alt_path
df = pd.read_csv(dataset_path)

# 2. Data Exploration & Visualization
print("Generating Class Distribution Plot...")
plt.figure(figsize=(10, 6))
sns.countplot(y='Navigation_Action', data=df, order=df['Navigation_Action'].value_counts().index)
plt.title('Class Distribution of Navigation Action')
plt.xlabel('Count')
plt.ylabel('Navigation Action')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=150)
plt.close()

print("Generating Correlation Heatmap...")
plt.figure(figsize=(16, 12))
numerical_cols = df.select_dtypes(include=[np.number]).columns
sns.heatmap(df[numerical_cols].corr(), annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Features')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150)
plt.close()

print("Generating Feature Boxplots...")
key_features = ['Altitude', 'Velocity', 'LiDAR_Front', 'Depth_Mean']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, feature in enumerate(key_features):
    row, col = divmod(i, 2)
    sns.boxplot(x='Navigation_Action', y=feature, data=df, ax=axes[row, col])
    axes[row, col].set_title(f'Distribution of {feature}')
    axes[row, col].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_boxplots.png'), dpi=150)
plt.close()

# 3. Data Preprocessing
print("Preprocessing Data...")
X = df.drop('Navigation_Action', axis=1)
y = df['Navigation_Action']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# 4. Train Models
models = {
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Deep Neural Network (DNN)': MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, random_state=42)
}

results = []

for name, model in models.items():
    print(f"\\n--- Training {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(f"{name} Classification Report:\\n{report}")
    
    with open(os.path.join(output_dir, f'report_{name.replace(" ", "_")}.txt'), 'w') as f:
        f.write(report)
        
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Macro Precision': precision,
        'Macro Recall': recall,
        'Macro F1': f1,
        'Weighted F1': f1_w
    })
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cm_{name.replace(" ", "_")}.png'), dpi=150)
    plt.close()

# Feature Importance
print("Generating Feature Importance...")
rf_model = models['Random Forest']
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150)
plt.close()

print("Generating Feature Scatter for top 3 features...")
top_3_features = [X.columns[i] for i in indices[:3]]
plt.figure(figsize=(10, 8))
sns.pairplot(df, vars=top_3_features, hue='Navigation_Action')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_scatter.png'), dpi=150)
plt.close()

# 5. Comparative Analysis
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, 'comparative_results.csv'), index=False)

print("\\n--- Comparative Results ---")
print(results_df.to_string(index=False))

plt.figure(figsize=(10, 6))
sns.barplot(x='Accuracy', y='Model', data=results_df.sort_values(by='Accuracy', ascending=False))
plt.title('Model Accuracy Comparison')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150)
plt.close()

print("\\nAll tasks completed successfully!")
