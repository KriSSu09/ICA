import pandas as pd
from naive_bayes import CustomNaiveBayes
from cross_validate import cross_validate
from statistical_analysis import mean_confidence_interval


def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Drop the 'customerID' column as it is not useful for classification
    data.drop(columns=['customerID'], inplace=True)

    # Convert 'TotalCharges' and 'MonthlyCharges' to numeric and handle non-numeric entries
    for col in ['TotalCharges', 'MonthlyCharges']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        data[col].fillna(data[col].median(), inplace=True)

    # Handle missing values for other columns
    for col in data.columns:
        if data[col].dtype == 'object' and col != 'Churn':
            data[col].fillna(data[col].mode()[0], inplace=True)

    # Map binary columns ('Yes', 'No') to 1 and 0
    binary_map = {'Yes': 1, 'No': 0}
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        data[col] = data[col].map(binary_map)

    # One-hot encode non-binary categorical variables
    non_binary_categorical_cols = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                                   'StreamingMovies', 'Contract', 'PaymentMethod']
    data = pd.get_dummies(data, columns=non_binary_categorical_cols)

    # Apply normalization only to continuous features
    continuous_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in continuous_cols:
        col_mean = data[col].mean()
        col_std = data[col].std()
        data[col] = (data[col] - col_mean) / col_std if col_std else data[col]

    # Convert binary and one-hot encoded features to integers
    binary_cols = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling']
    for col in data.columns:
        if col in binary_cols or col in non_binary_categorical_cols:
            data[col] = data[col].astype(int)

    # Ensure all numeric features are non-negative integers
    for col in data.columns:
        if data[col].dtype == 'int32' or data[col].dtype == 'float64':
            data[col] = data[col].clip(lower=0).astype(int)

    # Separate features and target variable
    X = data.drop('Churn', axis=1).values
    y = data['Churn'].values

    return X, y


def main():
    # Load and preprocess the data
    file_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    X, y = load_and_preprocess_data(file_path)

    # Create an instance of the model
    model = CustomNaiveBayes()

    # Perform cross-validation
    cv_results = cross_validate(model, X, y, cv=5)

    # Print the average performance metrics along with confidence intervals
    print("Cross-validation results with confidence intervals:")
    for metric, values in cv_results.items():
        mean, lower_bound, upper_bound = mean_confidence_interval(values)
        print(
            f"{metric.capitalize()}: Mean = {mean:.4f}, 95% Confidence Interval = [{lower_bound:.4f}, {upper_bound:.4f}]")


if __name__ == "__main__":
    main()
