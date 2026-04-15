import joblib
import numpy as np
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

# ==============================================================================
# SERVE.PY — Servidor de Inferência em Tempo Real (Real-Time API)
#
# Suas anotações descrevem esta estratégia:
# "Modelo exposto como microserviço HTTP, respondendo sob demanda em ms."
# "Quando usar: aprovação de crédito, detecção de fraude, chatbots,
#  qualquer fluxo interativo."
#
# O que este servidor faz:
#   POST /predict  → recebe features, retorna a espécie prevista
#   GET  /health   → health check (suas anotações: "implementar health check")
#   GET  /info     → metadados do modelo em produção
#
# Por que FastAPI e não Flask?
#   - Validação de dados automática via Pydantic (menos bugs)
#   - Tipagem nativa do Python
#   - Performance superior (Starlette + ASGI)
#   - Documentação automática em /docs (Swagger)
# ==============================================================================

# Logging estruturado
# Suas anotações: "observabilidade: logging estruturado, métricas de latência"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Nomes das classes do Iris (0, 1, 2 → texto legível)
CLASS_NAMES = {0: "setosa", 1: "versicolor", 2: "virginica"}


# ==============================================================================
# CARREGAMENTO DO MODELO NO STARTUP
#
# @asynccontextmanager lifespan = código que roda quando o servidor sobe e
# quando ele desce. Ideal para carregar artefatos pesados uma única vez.
#
# Analogia SRE: é como o init de um serviço — inicializa conexões e
# carrega configurações antes de aceitar tráfego.
#
# Por que não carregar no momento da requisição?
#   Carregar o modelo a cada request = latência de segundos por chamada.
#   Carregar uma vez no startup = latência de milissegundos por chamada.
# ==============================================================================
model  = None
scaler = None
model_metadata = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # -- startup --
    global model, scaler, model_metadata

    log.info("Iniciando servidor de inferência...")
    log.info("Carregando artefatos do modelo...")

    try:
        model  = joblib.load("model.joblib")
        scaler = joblib.load("scaler.joblib")
        model_metadata = {
            "model_type":    type(model).__name__,
            "n_estimators":  model.n_estimators,
            "max_depth":     model.max_depth,
            "n_features_in": model.n_features_in_,
            "classes":       list(CLASS_NAMES.values()),
        }
        log.info(f"Modelo carregado: {model_metadata['model_type']}")
        log.info(f"  n_estimators={model_metadata['n_estimators']}, max_depth={model_metadata['max_depth']}")
        log.info("Servidor pronto para receber requisições.")
    except FileNotFoundError as e:
        log.error(f"Artefato não encontrado: {e}")
        log.error("Execute 'python save_model.py' primeiro.")
        raise

    yield  # servidor rodando aqui

    # -- shutdown --
    log.info("Servidor encerrando. Liberando recursos.")


# ==============================================================================
# APLICAÇÃO FASTAPI
# ==============================================================================
app = FastAPI(
    title="Iris Classifier API",
    description="Classifica flores Iris em setosa, versicolor ou virginica.",
    version="1.0.0",
    lifespan=lifespan,
)


# ==============================================================================
# SCHEMA DE ENTRADA — Pydantic
#
# Pydantic valida automaticamente os dados que chegam na requisição.
# Se o cliente mandar um campo errado, a API retorna erro 422 com descrição
# clara — sem precisar escrever validação manual.
#
# Field(...) define o valor (... = obrigatório), descrição e limites.
# ==============================================================================
class IrisInput(BaseModel):
    sepal_length: float = Field(..., ge=0.0, le=20.0, description="Comprimento da sépala (cm)")
    sepal_width:  float = Field(..., ge=0.0, le=20.0, description="Largura da sépala (cm)")
    petal_length: float = Field(..., ge=0.0, le=20.0, description="Comprimento da pétala (cm)")
    petal_width:  float = Field(..., ge=0.0, le=20.0, description="Largura da pétala (cm)")

    @field_validator("sepal_length", "sepal_width", "petal_length", "petal_width")
    @classmethod
    def deve_ser_positivo(cls, v, info):
        if v < 0:
            raise ValueError(f"{info.field_name} não pode ser negativo")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "sepal_length": 5.1,
                "sepal_width":  3.5,
                "petal_length": 1.4,
                "petal_width":  0.2,
            }
        }
    }


class PredictionOutput(BaseModel):
    species:     str
    class_id:    int
    confidence:  float
    probabilities: dict[str, float]
    latency_ms:  float


# ==============================================================================
# ENDPOINT: GET /health
#
# Suas anotações: "Health check endpoint implementado (GET /)"
#
# Todo serviço em produção precisa de um endpoint de health check.
# É o que o load balancer e o Kubernetes usam para saber se o pod
# está saudável e pode receber tráfego (liveness + readiness probe).
#
# Retorna 200 se tudo ok, 503 se o modelo não estiver carregado.
# ==============================================================================
@app.get("/health", tags=["Operações"])
async def health_check():
    if model is None or scaler is None:
        log.warning("Health check falhou: modelo não carregado")
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    return {
        "status":  "healthy",
        "model":   model_metadata.get("model_type"),
        "message": "Pronto para inferência",
    }


# ==============================================================================
# ENDPOINT: GET /info
#
# Metadados do modelo em produção.
# Suas anotações: "cada versão carrega metadados (métricas, dataset, parâmetros)"
# ==============================================================================
@app.get("/info", tags=["Operações"])
async def model_info():
    return {
        "model_metadata": model_metadata,
        "input_features": [
            "sepal_length (cm)",
            "sepal_width (cm)",
            "petal_length (cm)",
            "petal_width (cm)",
        ],
        "output_classes": CLASS_NAMES,
    }


# ==============================================================================
# ENDPOINT: POST /predict
#
# O endpoint principal de inferência.
#
# Fluxo de uma requisição:
#   1. Cliente envia JSON com as 4 medidas da flor
#   2. Pydantic valida os campos automaticamente
#   3. Features passam pelo mesmo scaler usado no treino
#   4. Modelo retorna classe e probabilidades
#   5. Resposta JSON com predição + confiança + latência
#
# Suas anotações: "respondendo sob demanda em ms"
# ==============================================================================
@app.post("/predict", response_model=PredictionOutput, tags=["Inferência"])
async def predict(request: Request, data: IrisInput):
    t_inicio = time.perf_counter()

    # Montar array de features na mesma ordem do treino
    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]])

    # Aplicar o mesmo scaler do treino
    # CRÍTICO: sem isso o modelo recebe dados na escala errada
    features_scaled = scaler.transform(features)

    # Predição
    class_id     = int(model.predict(features_scaled)[0])
    probabilities = model.predict_proba(features_scaled)[0]
    species      = CLASS_NAMES[class_id]
    confidence   = float(probabilities[class_id])

    t_fim      = time.perf_counter()
    latency_ms = (t_fim - t_inicio) * 1000

    # Log estruturado de cada request — essencial para observabilidade
    log.info(
        f"PREDICT | ip={request.client.host} | "
        f"input={features[0].tolist()} | "
        f"prediction={species} | "
        f"confidence={confidence:.4f} | "
        f"latency={latency_ms:.2f}ms"
    )

    return PredictionOutput(
        species=species,
        class_id=class_id,
        confidence=confidence,
        probabilities={
            name: float(probabilities[i])
            for i, name in CLASS_NAMES.items()
        },
        latency_ms=round(latency_ms, 3),
    )


# ==============================================================================
# INICIALIZAÇÃO DIRETA
# Permite rodar com: python serve.py
# Em produção use: uvicorn serve:app --host 0.0.0.0 --port 8000 --workers 4
# ==============================================================================
if __name__ == "__main__":
    import uvicorn

    print()
    print("=" * 60)
    print("INICIANDO SERVIDOR DE INFERÊNCIA")
    print("=" * 60)
    print()
    print("  API:           http://localhost:8000")
    print("  Docs (Swagger): http://localhost:8000/docs")
    print("  Health check:  http://localhost:8000/health")
    print("  Info modelo:   http://localhost:8000/info")
    print()
    print("Para testar (em outro terminal):")
    print()
    print('  curl -X POST http://localhost:8000/predict \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}\'')
    print()
    print("  # Deve retornar: setosa (flor com pétalas pequenas)")
    print()
    print('  curl -X POST http://localhost:8000/predict \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"sepal_length":6.7,"sepal_width":3.0,"petal_length":5.2,"petal_width":2.3}\'')
    print()
    print("  # Deve retornar: virginica (flor com pétalas grandes)")
    print()
    print("Logs estruturados aparecerão aqui conforme as requests chegarem.")
    print("=" * 60)
    print()

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
