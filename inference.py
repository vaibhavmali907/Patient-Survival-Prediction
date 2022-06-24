import tensorflow as tf
import pickle
import pandas as pd
import joblib

model = tf.keras.models.load_model('model_tuned.h5')
ethnicity_encoder = joblib.load(open('ethnicity_encoder.pkl', 'rb'))
gender_encoder = joblib.load(open('gender_encoder.pkl', 'rb'))
hospital_admit_source_encoder = joblib.load(open('hospital_admit_source_encoder.pkl', 'rb'))
icu_admit_source_encoder = joblib.load(open('icu_admit_source_encoder.pkl', 'rb'))
icu_stay_type_encoder = joblib.load(open('icu_stay_type_encoder.pkl', 'rb'))
icu_type_encoder = joblib.load(open('icu_type_encoder.pkl', 'rb'))
apache_3j_bodysystem_encoder = joblib.load(open('apache_3j_bodysystem_encoder.pkl', 'rb'))
apache_2_bodysystem_encoder = joblib.load(open('apache_2_bodysystem_encoder.pkl', 'rb'))
s_imputer = pickle.load(open('s_imputer.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

def predict(df):
    df['ethnicity'] = ethnicity_encoder.transform(df['ethnicity'])
    df['gender'] = gender_encoder.transform(df['gender'])
    df['hospital_admit_source'] = hospital_admit_source_encoder.transform(df['hospital_admit_source'])
    df['icu_admit_source'] = icu_admit_source_encoder.transform(df['icu_admit_source'])
    df['icu_stay_type'] = icu_stay_type_encoder.transform(df['icu_stay_type'])
    df['icu_type'] = icu_type_encoder.transform(df['icu_type'])
    df['apache_3j_bodysystem'] = apache_3j_bodysystem_encoder.transform(df['apache_3j_bodysystem'])
    df['apache_2_bodysystem'] = apache_2_bodysystem_encoder.transform(df['apache_2_bodysystem'])
    df = s_imputer.transform(df)
    df = pd.DataFrame(scaler.transform(df))
    predictions = model.predict(df)
    output = predictions.astype(int)
    output = output.tolist()
    return output
