import numpy as np
import pandas as pd
from datetime import datetime

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
X = df.drop(["income"], axis=1)
y = df["income"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

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


model_results = {
                'Penalty': [],
                'C': [],
                'Solver': [],
                'Accuracy': []
}

classifier__penalty = ['l2']
classifier__solver = ['newton-cg', 'liblinear', 'sag']
classifier__C = np.logspace(-4, 4, 20)
classifier__n_estimators = list(range(10,101,10))
# classifier__max_features = list(range(6,32,5))}

MaxAccuracy = 0
MaxAccuracyC = -9999
MaxAccuracyN = 0
MaxAccuracyS = 0

for p in classifier__penalty:
    for c in classifier__C:
        for s in classifier__solver:
            # for m in classifier__max_features:

            # Run training
            logreg = LogisticRegression(penalty=p, C=c, solver=s)
            logreg.fit(X_train, y_train)

            # Run evaluation
            y_pred = logreg.predict(X_test)

            score = accuracy_score(y_test, y_pred)

            if score > MaxAccuracy:
                MaxAccuracy = score
                maxAccuracyC = int(c*10000)
                maxAccuracyS = s
                maxAccuracyP = p

            model_results['Penalty'].append(p)
            model_results['C'].append(str(int(c*10000)))
            model_results['Solver'].append(s)
            model_results['Accuracy'].append(score)
            customStep = int(c*10000)

r = pd.DataFrame()
r['Penalty'] = model_results['Penalty']
r['C'] = model_results['C']
r['Solver'] = model_results['Solver']
r['Accuracy'] = model_results['Accuracy']

r.to_csv("data/fine-tuning.csv", index=False)
