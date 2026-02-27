import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.preprocessing import RobustScaler
from dotenv import load_dotenv

load_dotenv()

USER = os.getenv('DB_USER')
PASSWORD = os.getenv('DB_PASSWORD')
HOST = os.getenv('DB_HOST')
DATABASE = os.getenv('DB_NAME')

def get_engine():
    return create_engine(f'mysql+pymysql://{USER}:{PASSWORD}@{HOST}/{DATABASE}')

def process_with_feature_engineering():
    engine = get_engine()
    
    print("1. Lendo dados de 'transactions'...")
    df = pd.read_sql("SELECT * FROM transactions", engine)

    print("2. Criando novas features...")
    
    df['hour'] = (df['Time'] % 86400) // 3600
    
    df['log_amount'] = np.log1p(df['Amount'])
    
    v_cols = ['V1', 'V3', 'V4', 'V7', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17']
    df['v_sum'] = df[v_cols].sum(axis=1)

    print("3. Escalonando features...")
    scaler = RobustScaler()
    
    df['std_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['std_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

    cols_to_drop = ['Time', 'Amount']
    df_train = df.drop(cols_to_drop, axis=1)

    cols = [c for c in df_train.columns if c != 'Class'] + ['Class']
    df_train = df_train[cols]

    print(f"4. Salvando em 'train_transactions' ({len(df_train)} linhas)...")
    df_train.to_sql('train_transactions', con=engine, if_exists='replace', index=False, chunksize=10000)
    print("Nova tabela de treino com Feature Engineering criada!")

if __name__ == "__main__":
    process_with_feature_engineering()