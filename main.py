from preprocess import load_data

import optuna
import dvc.api
import subprocess

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import joblib
import json
from datetime import datetime

# Load and split data
X, y = load_data(True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the objective function for Optuna optimization
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 1000, step=100)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    pipeline = Pipeline(
        [
            ('vectorizer', TfidfVectorizer()),
            ('model', RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf))
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# Create and run the Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=25)

# Get the best hyperparameters
best_params = study.best_params
best_accuracy = study.best_value

print('Best Parameters:', best_params)
print('Best Accuracy:', best_accuracy)

# Fit the pipeline with the best parameters on the entire dataset
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('model', RandomForestClassifier(**best_params))
])
pipeline.fit(X, y)
# Save model
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
model_filename = f'model/model_{timestamp}.joblib'
joblib.dump(pipeline, model_filename)

# Create a report
report = {
    'timestamp': timestamp,
    'best_params': best_params,
    'best_accuracy': best_accuracy,
    'data_version': dvc.api.get_url('data/Ethos_Dataset_Binary.csv'),
    'Model': 'Random Forest Classifier',
    'model_version': model_filename
}

# Save the report as a JSON file
report_filename = f'reports/report_{timestamp}.json'
with open(report_filename, 'w') as f:
    json.dump(report, f)


# DVC commands to track
subprocess.run(['dvc', 'add', report_filename])
subprocess.run(['dvc', 'add', model_filename])
subprocess.run(['dvc', 'commit', report_filename, model_filename])
# subprocess.run(['dvc', 'push'])
