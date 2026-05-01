import os
import pandas as pd
import xgboost as xgb

# Add these two lines at the top (outside the class)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

class DiseaseModel:
    def __init__(self):
        self.symptoms = None
        self.pred_disease = None
        self.model = xgb.XGBClassifier()
        self.diseases = self.disease_list(os.path.join(DATA_DIR, 'dataset.csv'))
    def predict(self, X):
       prediction = self.model.predict(X)
       prob = self.model.predict_proba(X)
       return prediction, prob    

    def load_xgboost(self, model_path):
        self.model.load_model(model_path)

    def disease_list(self, kaggle_dataset):
        df = pd.read_csv(os.path.join(DATA_DIR, 'clean_dataset.tsv'), sep='\t')
        # Preprocessing
        y_data = df.iloc[:,-1]
        X_data = df.iloc[:,:-1]

        self.all_symptoms = X_data.columns

        # Convert y to categorical values
        y_data = y_data.astype('category')
        
        return y_data.cat.categories
