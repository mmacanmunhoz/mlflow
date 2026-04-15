# ==============================================================================
# DOCKERFILE — Containerização do servidor de inferência
#
# Suas anotações:
# "Docker é o padrão para garantir reproducibilidade entre ambientes."
# "O container encapsula: código do serviço, artefato do modelo,
#  todas as dependências (Python version, libs)"
#
# Por que isso importa?
#   Sem Docker: "funciona na minha máquina" — problema clássico.
#   Com Docker: a imagem carrega TUDO que o serviço precisa.
#   dev, staging e prod rodam exatamente o mesmo binário.
#
# Estrutura desta imagem:
#   Base  → Python 3.13 slim (sem pacotes desnecessários, imagem menor)
#   Deps  → instala requirements.txt
#   App   → copia código e artefatos
#   Run   → inicia o FastAPI com uvicorn
# ==============================================================================

# Imagem base: python:3.13-slim
# "slim" = sem compiladores, docs e pacotes extras → imagem ~3x menor
FROM python:3.13-slim

# Metadados da imagem (boa prática)
LABEL description="Iris Classifier — Real-Time Inference API"
LABEL version="1.0.0"

# Variáveis de ambiente
# PYTHONDONTWRITEBYTECODE → não gera arquivos .pyc (desnecessário em container)
# PYTHONUNBUFFERED        → logs aparecem em tempo real (sem buffer)
#   Suas anotações: "logging estruturado" — sem isso os logs ficam presos no buffer
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Diretório de trabalho dentro do container
WORKDIR /app

# -----------------------------------------------------------------------
# ETAPA 1: Instalar dependências
# Copiamos SOMENTE o requirements.txt primeiro — antes do código.
# Por quê? Cada linha do Dockerfile é uma camada (layer) com cache.
# Se o requirements.txt não mudou, o Docker reutiliza o cache desta
# camada e pula o pip install (build muito mais rápido).
# Se você copiar o código junto, qualquer mudança de código invalida
# o cache do pip install e ele reinstala tudo.
# -----------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------
# ETAPA 2: Copiar código e artefatos do modelo
# Copiamos depois das dependências para aproveitar o cache acima.
# -----------------------------------------------------------------------
COPY serve.py       .
COPY model.joblib   .
COPY scaler.joblib  .

# -----------------------------------------------------------------------
# PORTA EXPOSTA
# Documentação: informa que o container escuta na porta 8000.
# Não abre a porta automaticamente — isso é feito no docker run -p.
# -----------------------------------------------------------------------
EXPOSE 8000

# -----------------------------------------------------------------------
# HEALTHCHECK
# Suas anotações: "Health check endpoint implementado"
# O Docker verifica a saúde do container periodicamente.
# Se /health retornar erro 3 vezes seguidas, o container é marcado
# como "unhealthy" — Kubernetes pode reiniciá-lo automaticamente.
# -----------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# -----------------------------------------------------------------------
# COMANDO DE INICIALIZAÇÃO
# uvicorn = servidor ASGI de alta performance para FastAPI
#   --host 0.0.0.0  → aceita conexões de qualquer IP (necessário no container)
#   --port 8000     → porta interna do container
#   --workers 2     → 2 processos paralelos (ajuste conforme CPU disponível)
#   --log-level warning → deixa os logs da aplicação visíveis sem ruído do uvicorn
# -----------------------------------------------------------------------
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--log-level", "warning"]
