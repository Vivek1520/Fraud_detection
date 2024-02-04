# Let's start the code by importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm


train_data = pd.read_csv("fraudTrain.csv")


test_data = pd.read_csv("fraudTest.csv")

combined_data = pd.concat([train_data, test_data], axis=0)


def extract_datetime_features(df):
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['hour_of_day'] = df['trans_date_trans_time'].dt.hour
    df.drop('trans_date_trans_time', axis=1, inplace=True)
    return df

combined_data = extract_datetime_features(combined_data)


columns_to_drop = ["first", "last", "job", "dob", "trans_num", "street"]
combined_data.drop(columns_to_drop, axis=1, inplace=True)


X_combined = combined_data.drop("is_fraud", axis=1)
y_combined = combined_data["is_fraud"]


label_encoder = LabelEncoder()
X_combined["merchant"] = label_encoder.fit_transform(X_combined["merchant"])
X_combined["category"] = label_encoder.fit_transform(X_combined["category"])


categorical_columns = ["gender", "city", "state"]
onehot_encoder = OneHotEncoder(drop="first", handle_unknown='ignore')
X_combined_categorical = onehot_encoder.fit_transform(X_combined[categorical_columns]).toarray()


scaler = StandardScaler()
X_combined_numeric = scaler.fit_transform(X_combined.drop(categorical_columns, axis=1))


X_combined_encoded = np.hstack((X_combined_numeric, X_combined_categorical))


X_train = X_combined_encoded[:len(train_data)]
X_test = X_combined_encoded[len(train_data):]
y_train = y_combined[:len(train_data)]
y_test = y_combined[len(train_data):]

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)


n_components = 100
ipca = IncrementalPCA(n_components=n_components)


for batch in tqdm(np.array_split(X_resampled, 10), desc="Applying Incremental PCA"):
    ipca.partial_fit(batch)


X_resampled_pca = ipca.transform(X_resampled)
X_test_pca = ipca.transform(X_test)


rf_classifier = RandomForestClassifier(random_state=42)


rf_classifier.fit(X_resampled_pca, y_resampled)


y_pred = rf_classifier.predict(X_test_pca)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)


print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{confusion}")
print(f"Classification Report:\n{report}")