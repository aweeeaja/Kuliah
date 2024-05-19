import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_excel('/content/gdrive/MyDrive/perceptron/MLP/BA_AirlineReviews_CL_excel.xlsx')

# Generate Data
# target = np.random.choice([1, 2], 100)
target = data[data['SeatType']]

# Split Data
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.5, random_state=42)

# Count the number of data in each class
train_data_class1 = train_data[train_target == 1]
train_data_class2 = train_data[train_target == 2]

test_data_class1 = test_data[test_target == 1]
test_data_class2 = test_data[test_target == 2]

print(f"Train Data Class 1: {len(train_data_class1)}")
print(f"Train Data Class 2: {len(train_data_class2)}")
print(f"Test Data Class 1: {len(test_data_class1)}")
print(f"Test Data Class 2: {len(test_data_class2)}")