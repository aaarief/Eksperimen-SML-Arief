import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

def load_data(path):
    """
    Load dataset from a CSV file.
    """
    return pd.read_csv(path, sep=';')

def preprocess_data(df):
    # 1. Drop 'duration' column
    if 'duration' in df.columns:
        df = df.drop(columns=['duration'])
    
    # 2. Handle duplicates
    df = df.drop_duplicates()
    
    # --- Feature Engineering pdays ---
    if 'pdays' in df.columns:
        df['previously_contacted'] = np.where(df['pdays'] == 999, 0, 1)
        df = df.drop(columns=['pdays'])
        
    # --- Capping Outliers pada 'campaign' ---
    if 'campaign' in df.columns:
        upper_limit = df['campaign'].quantile(0.99)
        df['campaign'] = np.where(df['campaign'] > upper_limit, upper_limit, df['campaign'])
    
    # 3. Separate features and target
    if 'y' in df.columns:
        X = df.drop(columns=['y'])
        y = df['y']
        
        # 4. Encode target variable
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        X = df
        y = None
    
    # 5. Identify columns types (Hybrid Strategy)
    ordinal_cols = ['education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week']
    
    education_order = ['illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'professional.course', 'university.degree', 'unknown']
    default_order = ['no', 'yes', 'unknown']
    housing_order = ['no', 'yes', 'unknown']
    loan_order    = ['no', 'yes', 'unknown']
    contact_order = ['telephone', 'cellular']
    month_order   = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    day_order     = ['mon', 'tue', 'wed', 'thu', 'fri']
    
    all_ordinal_categories = [
        education_order, default_order, housing_order, loan_order, 
        contact_order, month_order, day_order
    ]
    
    categorical_cols = ['job', 'marital', 'poutcome']
    
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols if col not in ordinal_cols and col not in categorical_cols]
    
    # 6. Preprocessing pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('yeo_johnson', PowerTransformer(method='yeo-johnson')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=all_ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat_nominal', categorical_transformer, categorical_cols),
            ('cat_ordinal', ordinal_transformer, ordinal_cols)
        ])
    
    # 7. Split data
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 8. Apply preprocessing
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # --- Handle Imbalance dengan SMOTE ---
        print("Before SMOTE counts:", np.bincount(y_train))
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
        print("After SMOTE counts:", np.bincount(y_train_resampled))
        
        # Get feature names
        cat_nominal_names = preprocessor.named_transformers_['cat_nominal']['onehot'].get_feature_names_out(categorical_cols)
        feature_names = list(numerical_cols) + list(cat_nominal_names) + list(ordinal_cols)
        
        # Convert to DataFrame
        X_train_df = pd.DataFrame(X_train_resampled, columns=feature_names)
        X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
        
        return X_train_df, X_test_df, y_train_resampled, y_test, preprocessor
    else:
        # If no target, just transform X
        X_processed = preprocessor.fit_transform(X)
        return X_processed, preprocessor

if __name__ == "__main__":
    import os
    
    # Path file disesuaikan dengan struktur folder repository
    # Asumsi script dijalankan dari root repository (standar GitHub Actions)
    input_path = 'Eksperimen_SML_Moch-Arief-Kresnanda/bank-additional-full_raw.csv'
    
    # Jika dijalankan manual dari dalam folder preprocessing, sesuaikan path
    if not os.path.exists(input_path):
        input_path = '../bank-additional-full_raw.csv'

    print(f"Memproses data dari: {input_path}")
    
    try:
        # Load data
        df = load_data(input_path)
        
        # Jalankan fungsi preprocessing
        X_train, X_test, y_train, y_test, _ = preprocess_data(df)
        
        print("Data processed successfully.")
        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)
        
        # Simpan hasil (Output)
        # Disimpan di folder Eksperimen_SML_... agar rapi
        output_dir = 'Eksperimen_SML_Moch-Arief-Kresnanda'
        
        # Jika folder output tidak ada (misal running dari dalam folder preprocessing), gunakan folder parent
        if not os.path.exists(output_dir) and os.path.exists('../bank-additional-full_raw.csv'):
             output_dir = '..'
        elif not os.path.exists(output_dir):
             output_dir = '.' # Fallback ke folder saat ini
            
        # Gabungkan target kembali ke dataframe agar mudah disimpan
        X_train['y'] = y_train
        X_test['y'] = y_test
        
        X_train.to_csv(f'{output_dir}/train_processed.csv', index=False)
        X_test.to_csv(f'{output_dir}/test_processed.csv', index=False)
        
        print(f"Preprocessing selesai! File tersimpan di folder: {output_dir}")
        
    except Exception as e:
        print(f"Terjadi error: {e}")
        print("Pastikan file dataset ada di lokasi yang benar.")
