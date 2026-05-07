import pandas as pd
import numpy as np
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

def prepare_symptoms_array(symptoms):
    df = pd.read_csv(os.path.join(DATA_DIR, 'clean_dataset.tsv'), sep='\t')
    '''
    Convert a list of symptoms to a ndim(X) (in this case 131) that matches the
    dataframe used to train the machine learning model

    Output:
    - X (np.array) = X values ready as input to ML model to get prediction
    '''
    symptoms_array = np.zeros((1,133))
    
    for symptom in symptoms:
        symptom_idx = df.columns.get_loc(symptom)
        symptoms_array[0, symptom_idx] = 1

    return symptoms_array
