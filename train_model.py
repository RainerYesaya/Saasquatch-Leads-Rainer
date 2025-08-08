import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv('data/Companies.csv')

# 2. Data Cleaning
df.columns = df.columns.str.strip()  
df = df.drop_duplicates()            

# Handle missing values
for col in ['industry', 'company_size', 'revenue_range', 'email_response', 'technologies']:
    df[col] = df[col].fillna("Unknown")
for col in ['website_visits', 'content_downloads', 'lead_score']:
    df[col] = df[col].fillna(0)

# 3. Target Variable
df['target'] = df['meeting_booked'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

# 4. Feature Engineering
df['tech_count'] = df['technologies'].apply(lambda x: len(x.split(',')) if pd.notnull(x) else 0)

# 5. Feature Selection
features = ['industry', 'company_size', 'revenue_range', 
            'website_visits', 'content_downloads', 'email_response', 'tech_count']
X = df[features]
y = df['target']

# 6. Label Encoding for categorical features
label_encoders = {}
for col in ['industry', 'company_size', 'revenue_range', 'email_response']:
    le = LabelEncoder()
    X.loc[:, col] = le.fit_transform(X[col])  # Gunakan .loc untuk menghindari warning
    label_encoders[col] = le

# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 8. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9. Predictions
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# 10. Evaluation Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


joblib.dump(model, 'models/lead_scoring_model.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')

print("✅ Model and encoders saved successfully!")
