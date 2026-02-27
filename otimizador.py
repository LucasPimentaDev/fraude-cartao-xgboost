import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, average_precision_score
import pandas as pd
import os


PASTA_SAIDA = "RESULTADOS_MODELO"


def otimizar_hyperparametros(X_train, y_train):
    print("--- Iniciando Busca de Hiperparametros (Otimizacao) ---")
    
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    param_dist = {
            
            'n_estimators': [400, 500], 
            'max_depth': [4, 5, 6], 
            'learning_rate': [0.05, 0.1],
            'subsample': [0.85, 0.9],
            'colsample_bytree': [0.8, 0.85],
            'gamma': [0, 0.1, 0.2],
            'reg_lambda': [1, 2, 5],
            'reg_alpha': [0],
            'scale_pos_weight': [pos_weight]
        }

    xgb = XGBClassifier(
        tree_method='hist',
        random_state=42,
        n_jobs=1 
    )

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scorer = make_scorer(average_precision_score, response_method='predict_proba')

    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=10,
        scoring=scorer,
        cv=skf,
        verbose=3,
        random_state=42,
        n_jobs=8
    )

    print("Iniciando fit... Verifique o uso de CPU.")
    random_search.fit(X_train, y_train)
    df_resultados = pd.DataFrame(random_search.cv_results_)
    colunas_interesse = [
    'param_n_estimators', 
    'param_max_depth', 
    'param_learning_rate', 
    'mean_test_score', 
    'std_test_score', 
    'mean_fit_time'
    ]
    tabela_performance = df_resultados[colunas_interesse].sort_values(by='mean_test_score', ascending=False)
    tabela_performance.to_csv(os.path.join(PASTA_SAIDA, "tuning_results.csv"), index=False)
    print("\n" + "="*80)
    print("RANKING DE PERFORMANCE DO TUNING (TOP 10 COMBINAÇÕES)")
    print("="*80)
    print(tabela_performance.to_string(index=False, formatters={'mean_test_score': '{:,.4f}'.format, 'mean_fit_time': '{:,.2f}s'.format}))
    print("="*80)

    print(f"Melhor Score (AUPRC): {random_search.best_score_:.4f}")
    return random_search.best_estimator_