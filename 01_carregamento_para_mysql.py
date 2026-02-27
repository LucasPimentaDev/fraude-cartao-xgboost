import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv


load_dotenv()

USER = os.getenv('DB_USER')
PASSWORD = os.getenv('DB_PASSWORD')
HOST = os.getenv('DB_HOST')
DATABASE = os.getenv('DB_NAME')

def get_engine():
    return create_engine(f'mysql+pymysql://{USER}:{PASSWORD}@{HOST}/{DATABASE}')

def carregar_tabela(file_path, table_name):
    if not os.path.exists(file_path):
        print(f"Arquivo nao encontrado: {file_path}")
        return

    print(f"Iniciando processamento sequencial: {table_name}")
    engine = get_engine()
    chunk_size = 20000 
    
    try:
        reader = pd.read_csv(file_path, chunksize=chunk_size)
        
        total_rows = 0
        for i, chunk in enumerate(reader):
            mode = 'replace' if i == 0 else 'append'
            
            chunk.to_sql(
                name=table_name, 
                con=engine, 
                if_exists=mode, 
                index=False
            )
            
            total_rows += len(chunk)
            if (i + 1) % 5 == 0:
                print(f"Tabela {table_name}: {total_rows} linhas ja inseridas...")

        print(f"Sucesso: {table_name} finalizada com {total_rows} linhas.")

    except Exception as e:
        print(f"Erro ao carregar a tabela {table_name}: {e}")

if __name__ == "__main__":
    base_path = 'data2/Credit Card Fraud Detection/'
    
    tarefas = [
        (os.path.join(base_path, 'creditcard.csv'), 'transactions')
    ]

    print("Iniciando processo de carga sequencial.")

    for arquivo, tabela in tarefas:
        carregar_tabela(arquivo, tabela)

    print("Processo completo. Todos os dados estao no MySQL.")

    print("\n" + "="*50)
    print("CARACTERÍSTICAS DA TABELA")
    print("="*50)

    try:
        engine = get_engine()
        colunas = pd.read_sql("DESCRIBE transactions",engine)
        df_estrutura = pd.read_sql("SELECT * FROM transactions LIMIT 5", engine)
        total_banco = pd.read_sql("SELECT COUNT(*) as total FROM transactions", engine).iloc[0]['total']
        print(f"\nTOTAL DE REGISTROS: {total_banco}")
        print(f"TOTAL DE COLUNAS: {len(colunas['Field'])}")
        
        print("\nEstrutura das Colunas (Tipos):")
        print(df_estrutura.dtypes.value_counts())
        print("Nomes das colunas")
        print(",".join(colunas['Field'].tolist()))
        print("\n")

    except Exception as e:
        print(f"Erro ao tentar extrair informações da tabela: {e}")