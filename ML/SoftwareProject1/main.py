import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, auc, precision_recall_curve, \
    average_precision_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load dataset
data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Handling Missing Values
# Assuming ' ' (empty space) is used for missing values in 'TotalCharges'
data['TotalCharges'] = data['TotalCharges'].replace(' ', float('nan'))
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])
data.dropna(inplace=True)  # Removing rows with missing values

# Encoding Categorical Variables and Feature Scaling
categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                        'PaymentMethod']
numerical_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

# Creating transformers for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Preprocessing the data
X = data.drop('Churn', axis=1)
y = data['Churn'].map({'Yes': 1, 'No': 0})  # Encoding the target variable

X = preprocessor.fit_transform(X)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression Model
log_reg = LogisticRegression()
log_reg_cv_results = cross_validate(log_reg, X_train, y_train, cv=5,
                                    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
log_reg_auc = roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])

# Random Forest Model
random_forest = RandomForestClassifier()
rf_cv_results = cross_validate(random_forest, X_train, y_train, cv=5,
                               scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
rf_auc = roc_auc_score(y_test, random_forest.predict_proba(X_test)[:, 1])

# Model Evaluation
print("Logistic Regression Performance Metrics:")
print(classification_report(y_test, y_pred_log_reg))
print("Logistic Regression AUC-ROC:", log_reg_auc)
print("\nRandom Forest Performance Metrics:")
print(classification_report(y_test, y_pred_rf))
print("Random Forest AUC-ROC:", rf_auc)

# Comparative Analysis
print("\nLogistic Regression Cross-Validation Metrics:")
for metric, score in log_reg_cv_results.items():
    print(f"{metric}: {np.mean(score)}")

print("\nRandom Forest Cross-Validation Metrics:")
for metric, score in rf_cv_results.items():
    print(f"{metric}: {np.mean(score)}")

# Plotting results
# For Logistic Regression
fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, log_reg.predict_proba(X_test)[:, 1])
roc_auc_log_reg = auc(fpr_log_reg, tpr_log_reg)

precision_log_reg, recall_log_reg, _ = precision_recall_curve(y_test, log_reg.predict_proba(X_test)[:, 1])
average_precision_log_reg = average_precision_score(y_test, log_reg.predict_proba(X_test)[:, 1])

# For Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, random_forest.predict_proba(X_test)[:, 1])
roc_auc_rf = auc(fpr_rf, tpr_rf)

precision_rf, recall_rf, _ = precision_recall_curve(y_test, random_forest.predict_proba(X_test)[:, 1])
average_precision_rf = average_precision_score(y_test, random_forest.predict_proba(X_test)[:, 1])

plt.figure()
plt.plot(fpr_log_reg, tpr_log_reg, label='Logistic Regression (area = %0.2f)' % roc_auc_log_reg)
plt.plot(fpr_rf, tpr_rf, label='Random Forest (area = %0.2f)' % roc_auc_rf)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
# Plotting ROC Curve
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.show()

# Plotting Precision-Recall Curve
plt.figure()
plt.plot(recall_log_reg, precision_log_reg, label='Logistic Regression (AP = %0.2f)' % average_precision_log_reg)
plt.plot(recall_rf, precision_rf, label='Random Forest (AP = %0.2f)' % average_precision_rf)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="best")
plt.savefig('precision_recall_curve.png')
plt.show()
