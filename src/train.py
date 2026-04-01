import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import sys
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

input_dir = sys.argv[1]
output_dir = sys.argv[2]

os.makedirs(output_dir, exist_ok=True)

train_df = pd.read_csv(os.path.join(input_dir, 'train.csv'))
test_df  = pd.read_csv(os.path.join(input_dir, 'test.csv'))

X_train = train_df.drop('target', axis=1).values
y_train = train_df['target'].values
X_test  = test_df.drop('target', axis=1).values
y_test  = test_df['target'].values

mlflow.set_experiment("Heart_Disease_DVC_Pipeline")

with mlflow.start_run():
    mlflow.set_tag("model_type", "RandomForest")
    mlflow.set_tag("pipeline", "DVC")

    n_estimators = 100
    max_depth    = 15

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth",    max_depth)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))
    test_f1   = f1_score(y_test, model.predict(X_test))

    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy",  test_acc)
    mlflow.log_metric("test_f1",        test_f1)

    print(f"Train accuracy : {train_acc:.4f}")
    print(f"Test  accuracy : {test_acc:.4f}")
    print(f"Test  F1       : {test_f1:.4f}")

    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    mlflow.log_artifact('confusion_matrix.png')
    os.remove('confusion_matrix.png')

    mlflow.sklearn.log_model(model, "random_forest_model")

    import joblib
    model_path = os.path.join(output_dir, 'model.pkl')
    joblib.dump(model, model_path)

    print("Done: model saved and logged to MLflow")