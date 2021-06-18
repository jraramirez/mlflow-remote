import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from joblib import dump

import warnings
warnings.filterwarnings('ignore')

# Load dataset
file = "data/adult_census.csv"
df = pd.read_csv(file, encoding="latin-1")

# Clean up Missing values
df[df == "?"] = np.nan
for col in ["workclass", "occupation", "native.country"]:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Split data to train-test
test_size = 0.3
X = df.drop(["income"], axis=1)
y = df["income"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

# Feature engineering
categorical = [
    "workclass",
    "education",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native.country",
]
for feature in categorical:
    le = preprocessing.LabelEncoder()
    X_train[feature] = le.fit_transform(X_train[feature])
    X_test[feature] = le.transform(X_test[feature])
    dump(le, "joblib/le." + feature + ".joblib")                            # Save the label encoders and re-use them during prediction

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Run training
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Run evaluation
y_pred = logreg.predict(X_test)

dump(logreg, "joblib/census_model.joblib")
dump(scaler, "joblib/scaler.joblib")                                        # Save the scaler and re-use them during prediction
