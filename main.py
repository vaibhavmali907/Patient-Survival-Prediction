# Data manipulation
import pickle

import numpy
import numpy as np
import pandas as pd

# Plotting and visualization
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

# Train-test split and k-fold cross validation
from sklearn.model_selection import train_test_split

# Missing data imputation
from sklearn.impute import SimpleImputer

# Categorical data encoding
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Deep learning
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

# Model evaluation
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Model loading
import keras_tuner as kt

# Warning suppression
import warnings
warnings.filterwarnings('ignore')

# Loading the data
data = pd.read_csv(r'Dataset.csv')

# Printing the dataframe
print(data)

# Shape of the data
print(pd.Series({"Shape of the dataset": data.shape}).to_string())

# Count of observations
print(pd.Series({"Number of observations in the dataset": len(data)}).to_string())

# Count of columns
print(pd.Series({"Total number of columns in the dataset": len(data.columns)}).to_string())

# Column names
print(data.columns)

# Column datatypes
print(data.dtypes)

# Count of column datatypes
cols_int = data.columns[data.dtypes == 'int64'].tolist()
cols_float = data.columns[data.dtypes == 'float64'].tolist()
cols_object = data.columns[data.dtypes == 'object'].tolist()
print(pd.Series({"Number of integer columns": len(cols_int),
                 "Number of float columns": len(cols_float),
                 "Number of object columns": len(cols_object)}).to_string())

# Count of duplicate rows
print(pd.Series({"Number of duplicate rows in the dataset": data.duplicated().sum()}).to_string())

# Count of duplicate columns
print(pd.Series({"Number of duplicate columns in the dataset": data.T.duplicated().sum()}).to_string())

# Statistical description of numerical variables in the dataset
data.describe()

# Statistical description of categorical variables in the dataset
data.describe(include = ['O'])

data.drop(['encounter_id','patient_id','hospital_id'], axis=1,inplace= True)
print(data)

# Label Encoder
ethnicity_encoder=LabelEncoder()
data['ethnicity'] = ethnicity_encoder.fit_transform(data['ethnicity'])
pickle.dump(ethnicity_encoder, open('ethnicity_encoder.pkl','wb'))

gender_encoder=LabelEncoder()
data['gender'] = gender_encoder.fit_transform(data['gender'])
pickle.dump(gender_encoder, open('gender_encoder.pkl','wb'))

hospital_admit_source_encoder=LabelEncoder()
data['hospital_admit_source'] = hospital_admit_source_encoder.fit_transform(data['hospital_admit_source'])
pickle.dump(hospital_admit_source_encoder, open('hospital_admit_source_encoder.pkl','wb'))

icu_admit_source_encoder=LabelEncoder()
data['icu_admit_source'] = icu_admit_source_encoder.fit_transform(data['icu_admit_source'])
pickle.dump(icu_admit_source_encoder, open('icu_admit_source_encoder.pkl','wb'))

icu_stay_type_encoder=LabelEncoder()
data['icu_stay_type'] = icu_stay_type_encoder.fit_transform(data['icu_stay_type'])
pickle.dump(icu_stay_type_encoder, open('icu_stay_type_encoder.pkl','wb'))

icu_type_encoder=LabelEncoder()
data['icu_type'] = icu_type_encoder.fit_transform(data['icu_type'])
pickle.dump(icu_type_encoder, open('icu_type_encoder.pkl','wb'))

apache_3j_bodysystem_encoder=LabelEncoder()
data['apache_3j_bodysystem'] = apache_3j_bodysystem_encoder.fit_transform(data['apache_3j_bodysystem'])
pickle.dump(apache_3j_bodysystem_encoder, open('apache_3j_bodysystem_encoder.pkl','wb'))

apache_2_bodysystem_encoder=LabelEncoder()
data['apache_2_bodysystem'] = apache_2_bodysystem_encoder.fit_transform(data['apache_2_bodysystem'])
pickle.dump(apache_2_bodysystem_encoder, open('apache_2_bodysystem_encoder.pkl','wb'))



X = data.drop(['hospital_death'], axis = 1) # Independent variables
y = data['hospital_death'] # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True)

# Training set - Predictor variables
print(X_train)

# Test set - Predictor variables
print(X_test)


# Simple Imputer

s_imputer = SimpleImputer(missing_values=numpy.nan,strategy='mean', verbose= 0)
s_imputer.fit(data.iloc[:,:])

X_train = pd.DataFrame(s_imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(s_imputer.transform(X_test), columns=X_test.columns)

pickle.dump(s_imputer,open('s_imputer.pkl','wb'))

# Min-max normalization of predictors in the training set
scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

pickle.dump(scaler, open('scaler.pkl','wb'))

# Adding layers to sequential model
model = tf.keras.Sequential()
model.add(Dense(16, input_shape = (182,)))
model.add(Dense(12, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(4, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()


# Specifying loss function and optimizer
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Training the model
history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 5, batch_size = 64)

# Visualization of model accuracy
model_accuracy = pd.DataFrame()
model_accuracy['accuracy'] = history.history['accuracy']
model_accuracy['val_accuracy'] = history.history['val_accuracy']

plt.figure(figsize = (9, 6))
sns.lineplot(data = model_accuracy['accuracy'], label = 'Train')
sns.lineplot(data = model_accuracy['val_accuracy'], label = 'Test')
plt.title('Model accuracy', fontsize = 14)
plt.ylabel('Accuracy', fontsize = 14)
plt.xlabel('Epoch', fontsize = 14)
plt.legend()
plt.show()

# Visualization of model loss
model_loss = pd.DataFrame()
model_loss['loss'] = history.history['loss']
model_loss['val_loss'] = history.history['val_loss']

plt.figure(figsize = (9, 6))
sns.lineplot(data = model_loss['loss'], label = 'Train')
sns.lineplot(data = model_loss['val_loss'], label = 'Test')
plt.title('Model loss', fontsize = 14)
plt.ylabel('Loss', fontsize = 14)
plt.xlabel('Epoch', fontsize = 14)
plt.legend()
plt.show()

# Prediction on test set
pred = model.predict(X_test)
threshold = 0.5
y_pred = [0 if pred[i][0] < threshold else 1 for i in range(len(pred))]

# Function to compute and visualize confusion matrix
def confusion_mat(y_pred, y_test):
    class_names = [0, 1]
    tick_marks_y = [0.5, 1.5]
    tick_marks_x = [0.5, 1.5]
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    confusion_matrix_df = pd.DataFrame(confusion_matrix, range(2), range(2))
    plt.figure(figsize = (6, 4.75))
    plt.title("Confusion Matrix", fontsize = 14)
    hm = sns.heatmap(confusion_matrix_df, annot = True, annot_kws = {"size": 16}, fmt = 'd') # font size
    hm.set_xlabel("Predicted label", fontsize = 14)
    hm.set_ylabel("True label", fontsize = 14)
    hm.set_xticklabels(class_names, fontdict = {'fontsize': 14}, rotation = 0, ha = "right")
    hm.set_yticklabels(class_names, fontdict = {'fontsize': 14}, rotation = 0, ha = "right")
    plt.grid(False)
    plt.show()

# Confusion matrix
confusion_mat(y_pred, y_test)

# Evaluation metrics
acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(pd.Series({"Accuracy": acc,
                 "ROC-AUC": roc_auc,
                 "Precision": precision,
                 "Recall": recall,
                 "F1-score": f1}).to_string())

# Building the model
def model_builder(ht):
    model = Sequential()
    model.add(keras.layers.Flatten(input_shape = (X_train.shape[1],)))

    # Tuning the number of units in the first Dense layer
    ht_units = ht.Int('units', min_value = 32, max_value = 512, step = 32) # 32-512
    model.add(keras.layers.Dense(units = ht_units, activation = 'relu'))
    model.add(keras.layers.Dense(12, activation = 'relu'))
    model.add(keras.layers.Dense(8, activation = 'relu'))
    model.add(keras.layers.Dense(4, activation = 'relu'))
    model.add(keras.layers.Dense(1, activation = 'sigmoid'))

    # Tuning the learning rate for the optimizer
    ht_learning_rate = ht.Choice('learning_rate', values = [0.01, 0.001, 0.0001])

    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    return model

# Making the tuner
tuner = kt.Hyperband(model_builder,
                     objective = 'val_accuracy',
                     max_epochs = 5,
                     factor = 3,
                     directory = 'dir_2')

# Early stopping
stop_early = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5)

# Implementing the tuner
tuner.search(X_train, y_train, epochs = 5, validation_split = 0.2, callbacks = [stop_early])

# Get the optimal hyperparameters
best_hparams = tuner.get_best_hyperparameters(num_trials = 1)[0]

print("-------- The hyperparameter search is complete --------")
print(" ")
print(pd.Series({"Optimal number of units in the first densely-connected layer": best_hparams.get('units'),
                 "Optimal learning rate for the optimizer": best_hparams.get('learning_rate')}).to_string())

# Building the model with optimal hyperparameters
model = tuner.hypermodel.build(best_hparams)

# Training the model
history = model.fit(X_train, y_train, epochs = 5, validation_split = 0.2)

# Validation accuracy
val_accuracy_optimal = history.history['val_accuracy']

# Computing best epoch in terms of maximum validation accuracy
best_epoch = val_accuracy_optimal.index(max(val_accuracy_optimal)) + 1
print(" ")
print(pd.Series({"Best epoch": (best_epoch)}).to_string())

# Re-instantiating the model
model_tuned = tuner.hypermodel.build(best_hparams)

# Re-training the hypermodel with the optimal number of epochs
model_tuned.fit(X_train, y_train, epochs = best_epoch, validation_split = 0.2)

# Evaluation on the test set
eval_tuned = model_tuned.evaluate(X_test, y_test)
print(" ")
print(pd.Series({"Test loss": eval_tuned[0],
                 "Test accuracy": eval_tuned[1]}).to_string())

# Confusion matrix
pred_tuned = model_tuned.predict(X_test)
threshold = 0.5
y_pred_tuned = [0 if pred_tuned[i][0] < threshold else 1 for i in range(len(pred_tuned))]
confusion_mat(y_pred_tuned, y_test)

# Evaluation metrics
acc = accuracy_score(y_test, y_pred_tuned)
roc_auc = roc_auc_score(y_test, y_pred_tuned)
precision = precision_score(y_test, y_pred_tuned)
recall = recall_score(y_test, y_pred_tuned)
f1 = f1_score(y_test, y_pred_tuned)

print(pd.Series({"Accuracy": acc,
                 "ROC-AUC": roc_auc,
                 "Precision": precision,
                 "Recall": recall,
                 "F1-score": f1}).to_string())

# Saving the model
model_tuned.save('model_tuned.h5')

# Loading the model
model_tuned_loaded = tf.keras.models.load_model('model_tuned.h5')

model_tuned_loaded.summary()

loss, acc=model_tuned_loaded.evaluate(X_test, y_test, verbose=2)

print(loss)

print(acc)

import pandas as pd

from inference import predict

df = pd.read_csv(r"test.csv")

output = predict(df)

print(output)

