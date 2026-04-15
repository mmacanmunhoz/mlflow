import numpy as np
import joblib
import mlflow
import time
from scipy import stats
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ==============================================================================
# MONITOR.PY — Monitoramento e Manutenção de Modelos em Produção
#
# Suas anotações:
# "Modelos de ML não são sistemas estáticos. Após o deploy, o desempenho
#  tende a degradar com o tempo."
#
# O que este script demonstra:
#   1. Simulação de produção ao longo do tempo (semanas)
#   2. Data Drift: detecção com KS Test e PSI
#   3. Model Drift: queda de métricas detectada automaticamente
#   4. Latência: percentis P50, P95, P99
#   5. Gatilho automático de retreino
#   6. Registro de avaliações no MLflow
# ==============================================================================

MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("iris-monitoring")

# Carrega modelo e scaler
iris = load_iris()

# Treina e salva baseline se não existir
try:
    model  = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
except FileNotFoundError:
    X, y = iris.data, iris.target
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train_sc, y_train)
    joblib.dump(model, "model.joblib")
    joblib.dump(scaler, "scaler.joblib")

# Dados de referência (treino original = "baseline")
X_ref, y_ref = iris.data, iris.target
feature_names = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"]


# ==============================================================================
# FUNÇÕES DE DETECÇÃO DE DRIFT
# ==============================================================================

def ks_test(referencia: np.ndarray, producao: np.ndarray, feature: str, alpha=0.05):
    """
    Kolmogorov-Smirnov Test — detecta mudança na distribuição de uma feature.

    Suas anotações: "KS test, PSI, Wasserstein, qui-quadrado por feature"

    Como funciona:
      Compara a distribuição cumulativa dos dados de referência (treino)
      com a distribuição dos dados de produção.
      p-valor < alpha → as distribuições são significativamente diferentes → DRIFT!

    p-valor = probabilidade de ver essa diferença por acaso se as distribuições
    fossem iguais. Muito baixo = diferença real, não acaso.
    """
    stat, p_value = stats.ks_2samp(referencia, producao)
    drift = p_value < alpha
    return {"feature": feature, "ks_stat": stat, "p_value": p_value, "drift": drift}


def psi(referencia: np.ndarray, producao: np.ndarray, bins=10):
    """
    Population Stability Index (PSI) — mede o quanto a distribuição mudou.

    Regra de bolso (suas anotações mencionam thresholds):
      PSI < 0.10  → estável, sem ação necessária
      PSI 0.10-0.25 → mudança moderada, monitorar
      PSI > 0.25  → mudança significativa, investigar/retreinar

    Vantagem sobre KS: dá uma magnitude da mudança, não só sim/não.
    """
    ref_hist, bins_edges = np.histogram(referencia, bins=bins)
    prod_hist, _         = np.histogram(producao, bins=bins_edges)

    # Evitar divisão por zero
    ref_pct  = (ref_hist  + 1e-6) / len(referencia)
    prod_pct = (prod_hist + 1e-6) / len(producao)

    psi_value = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    if psi_value < 0.10:
        status = "estavel"
    elif psi_value < 0.25:
        status = "atencao"
    else:
        status = "critico"
    return psi_value, status


def calcular_latencias(n=200):
    """Simula latências de inferência e calcula percentis."""
    latencias = []
    X_sample = X_ref[:n % len(X_ref) + 1]
    for i in range(n):
        x = X_ref[i % len(X_ref)].reshape(1, -1)
        t0 = time.perf_counter()
        scaler.transform(x)
        model.predict(scaler.transform(x))
        latencias.append((time.perf_counter() - t0) * 1000)
    return np.array(latencias)


def simular_producao(semana: int, drift_gradual=False, drift_severo=False):
    """
    Simula dados chegando em produção ao longo das semanas.

    Modela 3 cenários das suas anotações:
      Normal:       dados idênticos ao treino
      Drift gradual: mudança lenta (ex: equipamento envelhecendo)
      Drift severo:  mudança brusca (ex: novo equipamento, novo público)
    """
    np.random.seed(semana)
    n = 100
    X_prod = X_ref[np.random.choice(len(X_ref), n)].copy()

    if drift_gradual:
        # Adiciona ruído crescente semana a semana (drift gradual)
        fator = semana * 0.08
        X_prod[:, 0] += np.random.normal(fator, 0.2, n)   # sepal_len deriva
        X_prod[:, 2] += np.random.normal(fator * 0.5, 0.15, n)  # petal_len deriva

    if drift_severo:
        # Shift brusco em todas as features (ex: mudança de equipamento)
        X_prod += np.random.normal(1.5, 0.5, X_prod.shape)

    y_prod = y_ref[np.random.choice(len(y_ref), n)]
    return X_prod, y_prod


def avaliar_modelo(X_prod, y_prod):
    """Calcula métricas de qualidade preditiva com dados de produção."""
    X_scaled = scaler.transform(X_prod)
    y_pred   = model.predict(X_scaled)
    return {
        "accuracy":  accuracy_score(y_prod, y_pred),
        "f1":        f1_score(y_prod, y_pred, average="weighted"),
        "precision": precision_score(y_prod, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_prod, y_pred, average="weighted", zero_division=0),
    }


# ==============================================================================
# PARTE 1: MÉTRICAS DE QUALIDADE PREDITIVA SEMANA A SEMANA
# ==============================================================================
print("=" * 65)
print("PARTE 1: MONITORAMENTO DE QUALIDADE PREDITIVA")
print("=" * 65)
print()
print("Suas anotações: 'Modelos degradam com o tempo.'")
print("Simulando 8 semanas de produção com drift gradual:")
print()
print(f"  {'Semana':<8} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}  {'Status'}")
print(f"  {'-' * 68}")

THRESHOLD_ACC = 0.85   # suas anotações: "acurácia < 0.85 → aciona job de retreino"
retreino_necessario = False
semana_retreino = None

historico_metricas = []
for semana in range(1, 9):
    drift = semana > 4   # drift começa na semana 5
    X_prod, y_prod = simular_producao(semana, drift_gradual=drift)
    metricas = avaliar_modelo(X_prod, y_prod)
    historico_metricas.append(metricas)

    status = "ok"
    if metricas["accuracy"] < THRESHOLD_ACC and not retreino_necessario:
        status = "RETREINO NECESSARIO ←"
        retreino_necessario = True
        semana_retreino = semana

    print(
        f"  Semana {semana:<2} {metricas['accuracy']:>10.4f} {metricas['f1']:>10.4f} "
        f"{metricas['precision']:>10.4f} {metricas['recall']:>10.4f}  {status}"
    )

print()
print(f"  Threshold de retreino: accuracy < {THRESHOLD_ACC}")
if semana_retreino:
    print(f"  Gatilho disparado na semana {semana_retreino}!")
    print(f"  Analogia SRE: é como um alerta de SLO breach — a métrica cruzou o threshold.")


# ==============================================================================
# PARTE 2: DATA DRIFT — KS Test e PSI
#
# Suas anotações:
# "Data Drift: mudança na distribuição dos dados de entrada ao longo do tempo."
# "Causa degradação silenciosa e gradual."
# ==============================================================================
print()
print("=" * 65)
print("PARTE 2: DETECÇÃO DE DATA DRIFT (KS Test + PSI)")
print("=" * 65)
print()
print("Suas anotações:")
print("  'KS test, PSI, Wasserstein, qui-quadrado por feature'")
print()
print("Comparando semana 1 (sem drift) vs semana 8 (com drift severo):")
print()

X_sem1, _ = simular_producao(1, drift_gradual=False)
X_sem8, _ = simular_producao(8, drift_gradual=True)

print(f"--- KS Test (p-valor < 0.05 → drift detectado) ---")
print()
print(f"  {'Feature':<14} {'KS stat':>10} {'p-valor':>12} {'Drift?':>10}")
print(f"  {'-' * 50}")

drift_detectado = []
for i, feat in enumerate(feature_names):
    resultado = ks_test(X_ref[:, i], X_sem8[:, i], feat)
    drift_detectado.append(resultado["drift"])
    icone = "DRIFT ←" if resultado["drift"] else "ok"
    print(
        f"  {feat:<14} {resultado['ks_stat']:>10.4f} "
        f"{resultado['p_value']:>12.6f} {icone:>10}"
    )

print()
print(f"--- PSI (Population Stability Index) ---")
print()
print(f"  {'Feature':<14} {'PSI':>8}  {'Interpretação':<30}  {'Status'}")
print(f"  {'-' * 65}")

for i, feat in enumerate(feature_names):
    psi_val, status = psi(X_ref[:, i], X_sem8[:, i])
    if status == "estavel":
        rotulo = "PSI < 0.10 → sem acao necessaria"
    elif status == "atencao":
        rotulo = "PSI 0.10-0.25 → monitorar de perto"
    else:
        rotulo = "PSI > 0.25 → retreinar!"
    print(f"  {feat:<14} {psi_val:>8.4f}  {rotulo:<30}  {status}")

print()
print("Por que monitorar por feature e não só a accuracy?")
print("  Às vezes o drift começa antes da accuracy cair.")
print("  Detectar drift cedo = intervir antes do impacto no negócio.")
print("  Analogia SRE: monitorar CPU e memória antes de esperar o timeout.")


# ==============================================================================
# PARTE 3: MONITORAMENTO DE LATÊNCIA — PERCENTIS
#
# Suas anotações:
# "P50 (mediana), P90, P95, P99 — cada um revela um nível de problema"
# "P50: tempo típico | P95/P99: 'pior caso'"
# ==============================================================================
print()
print("=" * 65)
print("PARTE 3: MONITORAMENTO DE LATÊNCIA (Percentis)")
print("=" * 65)
print()
print("Calculando latências reais de inferência (200 requests simuladas)...")
print()

latencias = calcular_latencias(200)

p50  = np.percentile(latencias, 50)
p90  = np.percentile(latencias, 90)
p95  = np.percentile(latencias, 95)
p99  = np.percentile(latencias, 99)
pmax = latencias.max()

SLA_P95 = 50.0   # ms — limite de SLA definido

print(f"  {'Percentil':<12} {'Latência (ms)':>15}  {'Significado'}")
print(f"  {'-' * 65}")
print(f"  P50 (mediana) {p50:>15.3f}  tempo típico da maioria dos requests")
print(f"  P90           {p90:>15.3f}  90% dos requests ficam abaixo desse valor")
print(f"  P95           {p95:>15.3f}  apenas 5% dos requests excedem esse valor")
print(f"  P99           {p99:>15.3f}  apenas 1% excedem — detecta gargalos raros")
print(f"  Max           {pmax:>15.3f}  pior caso absoluto (outlier)")

print()
print(f"  SLA definido para P95: {SLA_P95:.0f} ms")
if p95 > SLA_P95:
    print(f"  STATUS: SLA VIOLADO no P95! ({p95:.2f}ms > {SLA_P95:.0f}ms) ← alerta ops")
else:
    print(f"  STATUS: P95 dentro do SLA ✓ ({p95:.2f}ms < {SLA_P95:.0f}ms)")

print()
print("Por que P99 importa?")
print("  A média esconde os piores casos. Se 1% dos requests demoram 500ms,")
print("  1 em cada 100 usuários tem péssima experiência — você só descobre")
print("  pelos percentis altos, nunca pela média.")
print("  Analogia SRE: é exatamente como você já monitora latência de APIs.")


# ==============================================================================
# PARTE 4: CONCEPT DRIFT vs DATA DRIFT
#
# Suas anotações:
# "Model Drift (Concept Drift): mudança na relação fundamental entre entradas
#  e a variável alvo."
# "Exemplo: modelo de fraude falha quando fraudadores mudam de tática."
# ==============================================================================
print()
print("=" * 65)
print("PARTE 4: DATA DRIFT vs CONCEPT DRIFT — a diferença")
print("=" * 65)
print()
print("  DATA DRIFT    → os dados de ENTRADA mudam")
print("                  Ex: novo equipamento de medição → features diferentes")
print("                  Detecção: KS Test, PSI nas features")
print()
print("  CONCEPT DRIFT → a RELAÇÃO entre feature e target muda")
print("                  Ex: fraudadores mudam de tática → o padrão muda")
print("                  As features são iguais, mas o significado mudou")
print("                  Detecção: queda de accuracy/F1 sem drift nas features")
print()
print("Simulando concept drift no Iris:")

# Concept drift: invertemos os labels de 2 classes (simula mudança de definição)
X_concept = X_ref.copy()
y_concept  = y_ref.copy()
y_concept[y_concept == 1] = 99   # placeholder
y_concept[y_concept == 2] = 1
y_concept[y_concept == 99] = 2   # swap versicolor ↔ virginica

metricas_ref     = avaliar_modelo(X_ref, y_ref)
metricas_concept = avaliar_modelo(X_concept, y_concept)

# KS test nas features
ks_features = [ks_test(X_ref[:, i], X_concept[:, i], feature_names[i]) for i in range(4)]
drift_features = any(r["drift"] for r in ks_features)

print()
print(f"  Accuracy SEM concept drift: {metricas_ref['accuracy']:.4f}")
print(f"  Accuracy COM concept drift: {metricas_concept['accuracy']:.4f}  ← caiu!")
print(f"  KS Test detectou drift nas features? {'Sim' if drift_features else 'Nao'}")
print()
print("  Conclusão: accuracy caiu MAS as features são iguais.")
print("  Isso é concept drift — a solução é retreinar com dados rotulados novos,")
print("  não apenas coletar mais dados com a mesma distribuição.")


# ==============================================================================
# PARTE 5: ESTRATÉGIAS DE RETREINO
# ==============================================================================
print()
print("=" * 65)
print("PARTE 5: QUANDO E COMO RETREINAR")
print("=" * 65)
print()
print("Suas anotações listam 5 estratégias:")
print()
print("  1. AGENDAMENTO PERIÓDICO")
print("     → Retreino fixo (semanal, mensal)")
print("     → Simples, previsível")
print("     → Pode retreinar desnecessariamente (custo) ou tarde demais")
print()
print("  2. GATILHO POR DEGRADAÇÃO DE PERFORMANCE")
print("     → Retreina quando accuracy cai abaixo do threshold")
print(f"     → Implementado acima: threshold={THRESHOLD_ACC}")
print("     → Requer feedback de rótulo rápido (nem sempre possível)")
print()
print("  3. GATILHO POR DATA DRIFT")
print("     → Retreina quando PSI > 0.25 ou KS p-valor < 0.01")
print("     → Proativo: age antes da accuracy cair")
print("     → Drift nem sempre = queda de performance (cuidado com alarmes)")
print()
print("  4. ATUALIZAÇÃO CONTÍNUA (online learning)")
print("     → Modelo atualiza a cada novo dado em tempo real")
print("     → Complexo, risco de instabilidade")
print("     → Usado em recomendação, detecção de anomalia")
print()
print("  5. RETREINO MANUAL")
print("     → Equipe decide quando retreinar (eventos extraordinários)")
print("     → Ex: COVID-19 invalidou todos os modelos de demanda de 2020")
print()
print("  BOA PRATICA das suas anotações:")
print("  'combinar agendamento regular + monitoramento de triggers'")


# ==============================================================================
# PARTE 6: REGISTRAR AVALIAÇÕES NO MLFLOW
#
# Cada semana de avaliação vira uma run no MLflow.
# Você abre localhost:5000 e vê o gráfico de degradação ao longo do tempo.
# ==============================================================================
print()
print("=" * 65)
print("PARTE 6: REGISTRANDO AVALIAÇÕES NO MLFLOW")
print("=" * 65)
print()
print("Registrando 8 semanas de monitoramento como runs MLflow...")
print("(cada semana = uma run separada no experimento 'iris-monitoring')")
print()

for semana, metricas in enumerate(historico_metricas, start=1):
    drift = semana > 4
    X_prod_week, _ = simular_producao(semana, drift_gradual=drift)
    psi_vals = [psi(X_ref[:, i], X_prod_week[:, i])[0] for i in range(4)]

    with mlflow.start_run(run_name=f"semana_{semana:02d}"):
        mlflow.log_metric("accuracy",  metricas["accuracy"])
        mlflow.log_metric("f1_score",  metricas["f1"])
        mlflow.log_metric("precision", metricas["precision"])
        mlflow.log_metric("recall",    metricas["recall"])
        mlflow.log_metric("psi_sepal_len", psi_vals[0])
        mlflow.log_metric("psi_petal_len", psi_vals[2])
        mlflow.log_metric("semana",    semana)
        mlflow.log_param("drift_ativo", drift)
        mlflow.log_param("retreino",    semana == semana_retreino)

    print(f"  Semana {semana}: accuracy={metricas['accuracy']:.4f}  psi_petal_len={psi_vals[2]:.4f}  {'← retreino!' if semana == semana_retreino else ''}")

print()
print(f"  Abra {MLFLOW_TRACKING_URI} → experimento 'iris-monitoring'")
print("  Clique em 'accuracy' para ver o gráfico de degradação ao longo das semanas.")
print("  Esse é o 'MLflow como sistema de observabilidade para modelos'")
print("  — exatamente como suas anotações descrevem.")

print()
print("=" * 65)
print("RESUMO: CHECKLIST DAS SUAS ANOTAÇÕES")
print("=" * 65)
print()
checks = [
    ("Logging estruturado por requisição",         True,  "serve.py — cada request logada com timestamp, payload e latência"),
    ("Métricas de qualidade preditiva",            True,  "accuracy, F1, precision, recall monitoradas por semana"),
    ("Detecção de data drift (KS Test + PSI)",     True,  "ks_test() e psi() implementados acima"),
    ("Threshold de retreino configurado",          True,  f"accuracy < {THRESHOLD_ACC} → gatilho dispara"),
    ("Latência monitorada com percentis",          True,  "P50, P90, P95, P99 calculados"),
    ("MLflow tracking das avaliações",             True,  "cada semana registrada como run no MLflow"),
    ("Estratégia de retreino documentada",         True,  "5 estratégias apresentadas acima"),
    ("Prometheus + Grafana",                       False, "próximo passo: expor /metrics no serve.py"),
    ("Pipeline de retreino automatizado",          False, "próximo passo: Airflow ou GitHub Actions"),
    ("Canary/shadow deploy",                       False, "próximo passo: MLflow Model Registry aliases"),
]

for item, feito, detalhe in checks:
    icone = "✓" if feito else "○"
    print(f"  [{icone}] {item}")
    print(f"       {detalhe}")
print()
print("=" * 65)
