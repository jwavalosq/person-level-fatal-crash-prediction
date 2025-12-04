---
output:
  pdf_document: default
  html_document: default
---
FARS2023 PERSON-LEVEL FATALITY PREDICTION
============================================================================
##Author: Jesus W Avalos Quizhpi
###Date: December 2025
============================================================================

Goals:
1. Run Random Forest, LightGBM, and XGBoost models to predict fatalities among participants in Motor-vehicle crashes using FARS 2023 datasets.
3. Reports F1-Score, ROC-AUC, and Confusion Matrix for each model
5. Saves comprehensive results to Google Drive


```python
# Install required packages
!pip install -q scikit-learn pandas numpy matplotlib seaborn lightgbm xgboost optuna joblib
```

    [?25l   [90mbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb[0m [32m0.0/404.7 kB[0m [31m?[0m eta [36m-:--:--[0m
[2K   [91mbbbbbbbbbbbbbbbbbbbbbbbbbb[0m[90mb:[0m[90mbbbbbbbbbbbbb[0m [32m266.2/404.7 kB[0m [31m7.8 MB/s[0m eta [36m0:00:01[0m
[2K   [90mbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb[0m [32m404.7/404.7 kB[0m [31m4.4 MB/s[0m eta [36m0:00:00[0m
    [?25h


```python
# ============================================================================
# 1: LIBRARIES & SETUP
# ============================================================================
import warnings
warnings.filterwarnings('ignore')

# Libraries
import random
import hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import joblib
from datetime import datetime


from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


import optuna
from optuna.samplers import TPESampler

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')


# Configure display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
```

    Mounted at /content/drive
    


```python
# ============================================================================
# 2: GENERATE 10 RANDOM SEEDS FROM INPUT STRING
# ============================================================================

def md5_hash(input_string):
    """Generate an MD5 hash from a given string
    Args:
      input_string: The string to hash.
    Returns:
      The MD5 hash as a hexadecimal string.
  """
    md5_hasher = hashlib.md5()
    md5_hasher.update(input_string.encode('utf-8'))
    return md5_hasher.hexdigest()

# Generate seeds from "PLANETEARTH"
input_string = "PLANETEARTH"
hashed_value = md5_hash(input_string)
seed_base = int(hashed_value, 16)

print(f"Input String: '{input_string}'")
print(f"MD5 Hash: {hashed_value}")
print(f"Base Seed: {seed_base}")

# Generate 10 random seeds
random.seed(seed_base)
RANDOM_SEEDS = [random.randint(0, 2**31-1) for _ in range(10)]

print(f"\n5 Random Seeds Generated:")
for i, seed in enumerate(RANDOM_SEEDS, 1):
    print(f"  Seed {i}: {seed}")


```

    Input String: 'PLANETEARTH'
    MD5 Hash: 891309412315c817d99554947406caac
    Base Seed: 182203076765220940304188608711018072748
    
    5 Random Seeds Generated:
      Seed 1: 640619102
      Seed 2: 1616531643
      Seed 3: 1886502974
      Seed 4: 915101968
      Seed 5: 1330353286
      Seed 6: 1238364284
      Seed 7: 2011418682
      Seed 8: 1354128445
      Seed 9: 1202834428
      Seed 10: 2054179773
    


```python
# Save seeds to Google Drive
#seed_output_path = '/content/drive/My Drive/FARS2023_seeds-PLANETEARTH-DEC2025.txt'
#with open(seed_output_path, 'w') as f:
#    f.write(f"FARS2023 Random Seeds - Generated: {datetime.now()}\n")
#    f.write(f"Input String: {input_string}\n")
#    f.write(f"MD5 Hash: {hashed_value}\n")
#    f.write(f"Base Seed: {seed_base}\n\n")
#    f.write("10 Random Seeds:\n")
#    for i, seed in enumerate(RANDOM_SEEDS, 1):
#        f.write(f"Seed {i}: {seed}\n")

#print(f"\nb Seeds saved to: {seed_output_path}\n")

```


```python
# ============================================================================
# 3: HELPER FUNCTIONS
# ============================================================================

def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*80}\n{title}\n{'='*80}\n")

def evaluate_model(y_true, y_pred, y_pred_proba):
    """Calculate all evaluation metrics"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        'ROC-AUC': roc_auc_score(y_true, y_pred_proba),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)
    }

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Survived', 'Died'],
                yticklabels=['Survived', 'Died'])

    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = 100 * cm[i, j] / total
            plt.text(j + 0.5, i + 0.7, f'({pct:.1f}%)',
                    ha='center', va='center', fontsize=9, color='gray')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  b Saved: {save_path}")
    plt.show()

    return cm

def to_binary(df, col, condition, default=0):
    """Convert column to binary based on condition"""
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        return condition(df[col]).astype(int)
    return default


```


```python
# ============================================================================
# 4: DATA LOADING
# ============================================================================

print_section("DATA LOADING")

from google.colab import files

print("Upload 4 CSV files: accident.csv, person.csv, vehicle.csv, weather.csv\n")
uploaded = files.upload()

print("\nLoading files...")
acc = pd.read_csv('accident.csv').drop_duplicates(subset='ST_CASE')
per = pd.read_csv('person.csv').drop_duplicates(subset=['ST_CASE', 'VEH_NO', 'PER_NO'])
veh = pd.read_csv('vehicle.csv').drop_duplicates(subset=['ST_CASE', 'VEH_NO'])
wx = pd.read_csv('weather.csv').drop_duplicates(subset='ST_CASE')

for df in [acc, per, veh, wx]:
    df.columns = df.columns.str.lower()

print(f"b Loaded: {len(acc):,} crashes, {len(per):,} persons, {len(veh):,} vehicles\n")

```

    
    ================================================================================
    DATA LOADING
    ================================================================================
    
    Upload 4 CSV files: accident.csv, person.csv, vehicle.csv, weather.csv



    Saving weather.csv to weather.csv
    Saving vehicle.csv to vehicle.csv
    Saving person.csv to person.csv
    Saving accident.csv to accident.csv
    
    Loading files...
    b Loaded: 37,654 crashes, 92,400 persons, 58,319 vehicles
    
    


```python
# Injury severity variable
print(per['inj_sev'].value_counts().sort_index())
print(per['inj_sevname'].value_counts())

```

    inj_sev
    0    24145
    1     6563
    2     9611
    3     9515
    4    40901
    5      195
    6        2
    9     1468
    Name: count, dtype: int64
    inj_sevname
    Fatal Injury (K)                40901
    No Apparent Injury (O)          24145
    Suspected Minor Injury (B)       9611
    Suspected Serious Injury (A)     9515
    Possible Injury (C)              6563
    Unknown/Not Reported             1468
    Injured, Severity Unknown         195
    Died Prior to Crash                 2
    Name: count, dtype: int64
    

Excluding the value b