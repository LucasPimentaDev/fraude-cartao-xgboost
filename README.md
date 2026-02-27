# Detecção de Fraudes em Cartão de Crédito com XGBoost

Este projeto implementa um pipeline de ponta a ponta para detecção de transações fraudulentas.

## 🛠️ Tecnologias e Recursos
- **Linguagem:** Python
- **Banco de Dados:** MySQL
- **Modelo:** XGBoost (Otimizado para 8 cores de CPU)
- **Métricas:** F1-Score e AUPRC

## 📈 Resultados Obtidos
Após o tuning de hiperparâmetros (30 fits), o modelo final atingiu:
- **AUPRC (Área sob a Curva Precision-Recall):** 0.8870
- **F1-Score (Classe Fraude):** 0.85
- **Recall:** 0.84 (Captura 84% das fraudes reais)
- **Precisão:** 0.87 (Baixa taxa de alarmes falsos)

## 🗂️ Estrutura do Projeto
1. `01_carregamento_para_mysql.py`: Carga dos dados CSV para MySQL.
2. `02_etl.py`: Criação de Features e escalonamentos.
3. `03_otimizador.py`: Busca de hiperparâmetros via RandomizedSearchCV.
4. `treino_xbg.py`: Script principal de treinamento e avaliação.# fraude-cartao-xgboost
