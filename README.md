# Iris ML Pipeline

Projeto de estudo do ciclo completo de Machine Learning: da análise exploratória ao monitoramento em produção. Usa o dataset Iris como base e MLflow para rastreamento de experimentos.

## Visão geral

```
eda.py              → análise exploratória (EDA)
preprocessing.py    → limpeza e preparação dos dados
model_selection.py  → comparação de modelos e busca de hiperparâmetros
train.py            → treino do modelo final com MLflow
save_model.py       → serialização do modelo e scaler para disco
serve.py            → API de inferência em tempo real (FastAPI)
monitor.py          → monitoramento em produção (drift, latência, retreino)
Dockerfile          → containerização do servidor de inferência
```

## Pré-requisitos

- Python 3.13+
- MLflow rodando localmente em `http://localhost:5000`

```bash
pip install -r requirements.txt
mlflow server --host 0.0.0.0 --port 5000
```

## Executando o pipeline

### 1. EDA — Análise Exploratória

```bash
python eda.py
```

Cobre os 4 pilares: visão geral dos dados, missing values, distribuição do target e outliers. Inclui análise de correlações entre features.

### 2. Preprocessing — Limpeza e Preparação

```bash
python preprocessing.py
```

Demonstra (com problemas injetados propositalmente):

- Imputação com mediana (numérico) e moda (categórico)
- Remoção de outliers severos via IQR × 3.0
- One-Hot Encoding
- Normalização com `StandardScaler`

### 3. Seleção de Modelos

```bash
python model_selection.py
```

Compara `LogisticRegression`, `DecisionTree` e `RandomForest` com `StratifiedKFold`. Demonstra overfitting vs underfitting. Roda Grid Search e Random Search para otimização de hiperparâmetros. Resultados registrados no experimento `iris-model-selection` do MLflow.

### 4. Treino

```bash
python train.py
```

Treina um `RandomForestClassifier` e registra parâmetros e métricas (accuracy, F1) no experimento `iris-classification` do MLflow.

### 5. Salvar Modelo

```bash
python save_model.py
```

Gera `model.joblib` e `scaler.joblib`. O scaler precisa ser salvo junto — ele foi ajustado nos dados de treino e deve aplicar exatamente a mesma transformação na inferência.

### 6. Servidor de Inferência

```bash
python serve.py
```

Ou com múltiplos workers:

```bash
uvicorn serve:app --host 0.0.0.0 --port 8000 --workers 4
```

**Endpoints:**

| Método | Rota       | Descrição                       |
|--------|------------|---------------------------------|
| GET    | `/health`  | Health check (liveness probe)   |
| GET    | `/info`    | Metadados do modelo em produção |
| POST   | `/predict` | Classificação de uma flor       |

**Exemplo de predição:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

Documentação interativa (Swagger) disponível em `http://localhost:8000/docs`.

### 7. Monitoramento

```bash
python monitor.py
```

Simula 8 semanas de produção com drift gradual. Cobre:

- Métricas semana a semana (accuracy, F1, precision, recall)
- Detecção de data drift com KS Test e PSI por feature
- Monitoramento de latência com percentis P50, P90, P95, P99
- Diferença entre data drift e concept drift
- Estratégias de retreino (agendamento, gatilho por degradação, gatilho por drift)

Resultados registrados no experimento `iris-monitoring` do MLflow.

## Docker

O `Dockerfile` containeriza apenas o servidor de inferência (`serve.py`). Requer os artefatos gerados pelo `save_model.py`.

```bash
# gerar os artefatos primeiro
python save_model.py

# build
docker build -t iris-classifier .

# run
docker run -p 8000:8000 iris-classifier
```

O container inclui healthcheck nativo (`GET /health`) compatível com Kubernetes liveness/readiness probes.

## MLflow UI

```
http://localhost:5000
```

Experimentos gerados pelo pipeline:

| Experimento           | Script             | O que registra                               |
|-----------------------|--------------------|----------------------------------------------|
| `iris-classification` | `train.py`         | params e métricas do treino final            |
| `iris-model-selection`| `model_selection.py` | CV accuracy de cada modelo/combinação      |
| `iris-monitoring`     | `monitor.py`       | métricas e PSI de cada semana de produção    |

## Dependências principais

| Pacote        | Versão  | Uso                              |
|---------------|---------|----------------------------------|
| scikit-learn  | 1.6.1   | modelos, pré-processamento, CV   |
| mlflow        | 2.21.3  | rastreamento de experimentos     |
| fastapi       | 0.115.12| servidor de inferência           |
| uvicorn       | 0.34.0  | ASGI server                      |
| pydantic      | 2.11.3  | validação de dados da API        |
| pandas        | 2.2.3   | manipulação de dados             |
| scipy         | 1.15.2  | KS test para detecção de drift   |
| joblib        | 1.4.2   | serialização de modelos          |


## Como instalar o MLFLOW

1 - Eu utilizei o kind para testar

```
helm uninstall mlflow -n mlflow || true

helm install mlflow community-charts/mlflow \
  -n mlflow \
  --create-namespace \
  --set backendStore.defaultSqlitePath=/mlflow/data/mlflow.db \
  --timeout 15m \
  --wait

```