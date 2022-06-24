import tensorflow as tf
import pickle
import pandas as pd
import joblib
import numpy as np

model = tf.keras.models.load_model('model_tuned.h5')
ethnicity_encoder = joblib.load(open('ethnicity_encoder.pkl', 'rb'))
gender_encoder = joblib.load(open('gender_encoder.pkl', 'rb'))
hospital_admit_source_encoder = joblib.load(open('hospital_admit_source_encoder.pkl', 'rb'))
icu_admit_source_encoder = joblib.load(open('icu_admit_source_encoder.pkl', 'rb'))
icu_stay_type_encoder = joblib.load(open('icu_stay_type_encoder.pkl', 'rb'))
icu_type_encoder = joblib.load(open('icu_type_encoder.pkl', 'rb'))
apache_3j_bodysystem_encoder = joblib.load(open('apache_3j_bodysystem_encoder.pkl', 'rb'))
apache_2_bodysystem_encoder = joblib.load(open('apache_2_bodysystem_encoder.pkl', 'rb'))
s_imputer = joblib.load(open('s_imputer.pkl','rb'))
scaler = joblib.load(open('scaler.pkl','rb'))
class_names = [1,0]


def predict(df):
    df = df[['age','bmi','elective_surgery','ethnicity','gender','height','hospital_admit_source','icu_admit_source','icu_id','icu_stay_type','icu_type','pre_icu_los_days',
 'readmission_status','weight','albumin_apache','apache_2_diagnosis','apache_3j_diagnosis','apache_post_operative','arf_apache','bilirubin_apache',
 'bun_apache','creatinine_apache','fio2_apache','gcs_eyes_apache','gcs_motor_apache','gcs_unable_apache','gcs_verbal_apache','glucose_apache','heart_rate_apache',
 'hematocrit_apache','intubated_apache','map_apache','paco2_apache','paco2_for_ph_apache',
 'pao2_apache','ph_apache','resprate_apache','sodium_apache','temp_apache','urineoutput_apache','ventilated_apache','wbc_apache','d1_diasbp_invasive_max','d1_diasbp_invasive_min',
 'd1_diasbp_max','d1_diasbp_min','d1_diasbp_noninvasive_max','d1_diasbp_noninvasive_min','d1_heartrate_max','d1_heartrate_min','d1_mbp_invasive_max','d1_mbp_invasive_min','d1_mbp_max','d1_mbp_min','d1_mbp_noninvasive_max','d1_mbp_noninvasive_min','d1_resprate_max',
 'd1_resprate_min','d1_spo2_max','d1_spo2_min','d1_sysbp_invasive_max','d1_sysbp_invasive_min','d1_sysbp_max','d1_sysbp_min','d1_sysbp_noninvasive_max','d1_sysbp_noninvasive_min', 'd1_temp_max','d1_temp_min',
 'h1_diasbp_invasive_max','h1_diasbp_invasive_min','h1_diasbp_max','h1_diasbp_min','h1_diasbp_noninvasive_max','h1_diasbp_noninvasive_min',
 'h1_heartrate_max','h1_heartrate_min','h1_mbp_invasive_max','h1_mbp_invasive_min','h1_mbp_max','h1_mbp_min','h1_mbp_noninvasive_max',
 'h1_mbp_noninvasive_min','h1_resprate_max','h1_resprate_min','h1_spo2_max','h1_spo2_min','h1_sysbp_invasive_max','h1_sysbp_invasive_min','h1_sysbp_max',
 'h1_sysbp_min','h1_sysbp_noninvasive_max','h1_sysbp_noninvasive_min','h1_temp_max','h1_temp_min','d1_albumin_max','d1_albumin_min',
 'd1_bilirubin_max','d1_bilirubin_min','d1_bun_max','d1_bun_min','d1_calcium_max','d1_calcium_min','d1_creatinine_max','d1_creatinine_min','d1_glucose_max','d1_glucose_min','d1_hco3_max','d1_hco3_min',
 'd1_hemaglobin_max','d1_hemaglobin_min','d1_hematocrit_max','d1_hematocrit_min','d1_inr_max',
 'd1_inr_min','d1_lactate_max','d1_lactate_min','d1_platelets_max','d1_platelets_min','d1_potassium_max','d1_potassium_min','d1_sodium_max','d1_sodium_min','d1_wbc_max',
 'd1_wbc_min','h1_albumin_max','h1_albumin_min','h1_bilirubin_max','h1_bilirubin_min','h1_bun_max','h1_bun_min','h1_calcium_max','h1_calcium_min','h1_creatinine_max','h1_creatinine_min','h1_glucose_max','h1_glucose_min','h1_hco3_max','h1_hco3_min','h1_hemaglobin_max',
 'h1_hemaglobin_min','h1_hematocrit_max','h1_hematocrit_min','h1_inr_max','h1_inr_min','h1_lactate_max','h1_lactate_min','h1_platelets_max','h1_platelets_min',
 'h1_potassium_max','h1_potassium_min','h1_sodium_max','h1_sodium_min','h1_wbc_max','h1_wbc_min','d1_arterial_pco2_max','d1_arterial_pco2_min','d1_arterial_ph_max','d1_arterial_ph_min','d1_arterial_po2_max','d1_arterial_po2_min','d1_pao2fio2ratio_max','d1_pao2fio2ratio_min','h1_arterial_pco2_max','h1_arterial_pco2_min',
 'h1_arterial_ph_max','h1_arterial_ph_min','h1_arterial_po2_max','h1_arterial_po2_min','h1_pao2fio2ratio_max','h1_pao2fio2ratio_min','apache_4a_hospital_death_prob','apache_4a_icu_death_prob','aids','cirrhosis',
 'diabetes_mellitus','hepatic_failure','immunosuppression','leukemia',
 'lymphoma','solid_tumor_with_metastasis','apache_3j_bodysystem','apache_2_bodysystem']]
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
    numpy_array = df.to_numpy()
    predictions = model.predict(numpy_array)
    output = predictions.astype(int)
    output = output.tolist()
    output = [class_names[class_predicted] for class_predicted in predictions]
    return output
