import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sqlalchemy import create_engine
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, PrecisionRecallDisplay
from dotenv import load_dotenv
from otimizador import otimizar_hyperparametros

load_dotenv()
USER, PASSWORD, HOST, DATABASE = os.getenv('DB_USER'), os.getenv('DB_PASSWORD'), os.getenv('DB_HOST'), os.getenv('DB_NAME')
engine = create_engine(f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}/{DATABASE}")

PASTA_IMG = "RESULTADOS_MODELO"

def garantir_pasta():
    if not os.path.exists(PASTA_IMG):
        os.makedirs(PASTA_IMG)
        print(f"Pasta {PASTA_IMG} criada.")


def exportar_artefatos(model):
    garantir_pasta()
    print("--- Exportando modelo treinado ---")
    

    model.save_model(os.path.join(PASTA_IMG, "modelo_fraud_xgb.json"))
    

    joblib.dump(model, os.path.join(PASTA_IMG, "modelo_final.pkl"))
    
    print(f"Modelo salvo na pasta {PASTA_IMG}")

def treinar_modelo_fraude():
    garantir_pasta()
    
    print("--- Lendo dados de treino do MySQL ---")
    df = pd.read_sql("SELECT * FROM train_transactions", engine)
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = otimizar_hyperparametros(X_train, y_train)

    print("--- Treinamento Final com Melhores Parâmetros ---")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)


    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]


    print("Salvando Matriz de Confusão...")
    plt.figure(figsize=(10, 7))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz de Confusão - XGBoost', fontsize=15)
    plt.ylabel('Realidade')
    plt.xlabel('Previsão do Modelo')
    plt.savefig(os.path.join(PASTA_IMG, "01_matriz_confusao.png"), dpi=300)
    plt.close()

    print("Salvando Curva Precision-Recall...")
    plt.figure(figsize=(10, 7))
    display = PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    display.ax_.set_title("Curva Precision-Recall (AUPRC)", fontsize=15)
    plt.savefig(os.path.join(PASTA_IMG, "02_curva_precision_recall.png"), dpi=300)
    plt.close()

    print("Salvando Feature Importance...")
    plt.figure(figsize=(12, 10))
    plot_importance(model, importance_type='gain', max_num_features=15, height=0.5,values_format="{v:.2f}")
    plt.title("Top 15 Variáveis Mais Importantes", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_IMG, "03_feature_importance.png"), dpi=300)
    plt.close()

    print("\n--- Relatório Final ---")
    print(classification_report(y_test, y_pred))
    auprc = average_precision_score(y_test, y_proba)
    print(f"Área sob a Curva Precision-Recall (AUPRC): {auprc:.4f}")
    
    return model

if __name__ == "__main__":

    modelo_final = treinar_modelo_fraude()
    
    exportar_artefatos(modelo_final)