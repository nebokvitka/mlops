import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

input_file = sys.argv[1]
output_dir = sys.argv[2]

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_file)

# Перевірка пропусків
df = df.dropna()

X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

train_df = pd.DataFrame(X_train, columns=X.columns)
train_df['target'] = y_train.values

test_df = pd.DataFrame(X_test, columns=X.columns)
test_df['target'] = y_test.values

train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

print("Done: {} train, {} test".format(len(train_df), len(test_df)))