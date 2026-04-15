import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np

# ==============================================================================
# CONFIGURAÇÃO DO MLFLOW
# MLflow é um sistema de rastreamento de experimentos. Pense nele como um
# "caderno de laboratório digital" que registra automaticamente: quais
# parâmetros você usou, quais métricas obteve e o modelo treinado em si.
# ==============================================================================
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("iris-classification")

print("=" * 60)
print("ETAPA 1: CARREGAMENTO DOS DADOS")
print("=" * 60)

# ==============================================================================
# CARREGAMENTO DOS DADOS
# O Iris Dataset é o "Hello World" do Machine Learning.
# São 150 flores de 3 espécies, cada uma com 4 medidas (features).
#
# X = as medidas de cada flor (150 linhas × 4 colunas)
# y = a espécie de cada flor (150 valores: 0, 1 ou 2)
# ==============================================================================
iris = load_iris()
X, y = iris.data, iris.target

print(f"Dataset carregado: {X.shape[0]} amostras, {X.shape[1]} features")
print(f"\nFeatures (o que medimos em cada flor):")
for i, nome in enumerate(iris.feature_names):
    print(f"  [{i}] {nome}")

print(f"\nClasses (o que queremos prever):")
for i, especie in enumerate(iris.target_names):
    qtd = np.sum(y == i)
    print(f"  Classe {i} = {especie:12} → {qtd} amostras")

print(f"\nPrimeiras 5 amostras do dataset (X):")
print(f"  {'Sepal Len':>10} {'Sepal Wid':>10} {'Petal Len':>10} {'Petal Wid':>10} {'Espécie':>12}")
for i in range(5):
    especie = iris.target_names[y[i]]
    print(f"  {X[i,0]:>10.1f} {X[i,1]:>10.1f} {X[i,2]:>10.1f} {X[i,3]:>10.1f} {especie:>12}")

print()
print("=" * 60)
print("ETAPA 2: DIVISÃO TREINO / TESTE")
print("=" * 60)

# ==============================================================================
# DIVISÃO TREINO / TESTE
# Não podemos testar o modelo com os mesmos dados que usamos para treinar
# (seria como dar a prova com as respostas). Por isso dividimos:
#
# - 80% dos dados → treino (o modelo aprende com eles)
# - 20% dos dados → teste (avaliamos se o modelo realmente aprendeu)
#
# random_state=42 garante que a divisão seja sempre igual, tornando o
# experimento reproduzível — qualquer pessoa rodando este código
# obtém exatamente os mesmos dados de treino e teste.
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Total de amostras: {len(X)}")
print(f"  → Treino: {len(X_train)} amostras ({len(X_train)/len(X)*100:.0f}%)")
print(f"  → Teste:  {len(X_test)} amostras ({len(X_test)/len(X)*100:.0f}%)")
print()
print("Por que fazer isso?")
print("  Se testarmos com os mesmos dados do treino, o modelo pode")
print("  ter 'decorado' as respostas sem realmente aprender — isso")
print("  se chama OVERFITTING. O conjunto de teste simula dados novos.")

print()
print("=" * 60)
print("ETAPA 3: DEFINIÇÃO DOS HIPERPARÂMETROS")
print("=" * 60)

# ==============================================================================
# HIPERPARÂMETROS
# São as configurações do modelo que você define ANTES do treino.
# Diferente dos parâmetros internos que o modelo aprende sozinho
# (pesos, limiares de decisão), estes você controla manualmente.
#
# n_estimators → quantas árvores de decisão a floresta vai ter
# max_depth    → profundidade máxima de cada árvore (quantas perguntas ela pode fazer)
# random_state → semente de aleatoriedade para reprodutibilidade
# ==============================================================================
params = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42,
}

print("Hiperparâmetros definidos:")
print(f"  n_estimators = {params['n_estimators']}")
print(f"    → A floresta terá {params['n_estimators']} árvores de decisão.")
print(f"    → Cada árvore vê uma parte diferente dos dados e vota na resposta.")
print(f"  max_depth    = {params['max_depth']}")
print(f"    → Cada árvore pode fazer no máximo {params['max_depth']} perguntas")
print(f"    → antes de chegar a uma decisão (limita complexidade).")
print(f"  random_state = {params['random_state']}")
print(f"    → Garante que a aleatoriedade interna do modelo seja sempre igual.")

print()
print("=" * 60)
print("ETAPA 4: TREINAMENTO DO MODELO")
print("=" * 60)

# ==============================================================================
# TREINAMENTO
# Random Forest = Floresta Aleatória.
# Cria N árvores de decisão (n_estimators), cada uma treinada em uma
# amostra aleatória dos dados. Na predição, todas as árvores votam
# e a classe mais votada é a resposta final.
#
# Por que é melhor que uma árvore só?
# Uma única árvore pode "decorar" os dados de treino (overfitting).
# Muitas árvores com perspectivas diferentes se compensam, resultando
# em um modelo mais robusto e generalizável.
#
# model.fit() = o momento em que o modelo aprende com os dados de treino.
# ==============================================================================
print("Treinando RandomForestClassifier...")
print(f"  Algoritmo: Random Forest (Floresta Aleatória)")
print(f"  Como funciona: cria {params['n_estimators']} árvores de decisão,")
print(f"  cada uma aprendendo com uma amostra diferente dos dados de treino.")
print(f"  Na predição, todas as árvores votam — ganha a classe mais votada.")

model = RandomForestClassifier(**params)
model.fit(X_train, y_train)

print(f"\n  Treino concluído!")
print(f"  O modelo agora conhece os padrões de {len(X_train)} flores.")

print()
print("=" * 60)
print("ETAPA 5: PREDIÇÃO E AVALIAÇÃO")
print("=" * 60)

# ==============================================================================
# PREDIÇÃO
# Agora usamos os dados de TESTE (que o modelo nunca viu) para avaliar
# se ele realmente aprendeu a classificar flores corretamente.
#
# y_pred = as previsões do modelo para cada flor do conjunto de teste
# y_test = as respostas corretas (gabarito)
# ==============================================================================
y_pred = model.predict(X_test)

print("Comparando previsões do modelo com o gabarito (primeiras 10):")
print(f"  {'#':>3} {'Previsto':>12} {'Real (gabarito)':>16} {'Acertou?':>10}")
for i in range(10):
    previsto = iris.target_names[y_pred[i]]
    real = iris.target_names[y_test[i]]
    acertou = "✓" if y_pred[i] == y_test[i] else "✗"
    print(f"  {i+1:>3} {previsto:>12} {real:>16} {acertou:>10}")

# ==============================================================================
# MÉTRICAS
# Accuracy (Acurácia): de todas as previsões, qual % acertou?
#   → Simples, mas pode enganar em datasets desbalanceados.
#
# F1 Score: média harmônica entre Precisão e Recall.
#   → Mais robusto. Penaliza tanto falsos positivos quanto negativos.
#   → average="weighted" leva em conta o tamanho de cada classe.
#   → Valores próximos de 1.0 são ótimos.
# ==============================================================================
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print()
print("Métricas de avaliação:")
print(f"  Accuracy (Acurácia): {acc:.4f} ({acc*100:.2f}%)")
print(f"    → O modelo acertou {acc*100:.1f}% das {len(y_test)} flores do conjunto de teste.")
print(f"  F1 Score (ponderado): {f1:.4f}")
print(f"    → Métrica mais robusta que considera erros em todas as classes.")

print()
print("Relatório detalhado por classe:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ==============================================================================
# MATRIZ DE CONFUSÃO
# Mostra visualmente onde o modelo erra.
# Linhas = classe real | Colunas = classe prevista
# A diagonal principal são os acertos. Fora da diagonal = erros.
# ==============================================================================
cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print("  (linhas = real, colunas = previsto)")
print(f"  {'':>12}", end="")
for nome in iris.target_names:
    print(f"  {nome:>12}", end="")
print()
for i, nome in enumerate(iris.target_names):
    print(f"  {nome:>12}", end="")
    for j in range(len(iris.target_names)):
        marcador = f"[{cm[i,j]}]" if i == j else f" {cm[i,j]} "
        print(f"  {marcador:>12}", end="")
    print()

print()
print("=" * 60)
print("ETAPA 6: REGISTRO NO MLFLOW")
print("=" * 60)

# ==============================================================================
# MLFLOW TRACKING
# Registra este experimento no MLflow para que você possa:
# - Comparar diferentes versões do modelo no futuro
# - Reproduzir exatamente este experimento
# - Acompanhar a evolução das métricas ao longo do tempo
#
# Acesse http://localhost:5000 no browser para ver a interface visual.
# ==============================================================================
with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    run_id = mlflow.active_run().info.run_id
    print(f"Experimento registrado no MLflow!")
    print(f"  Run ID:    {run_id}")
    print(f"  Servidor:  {MLFLOW_TRACKING_URI}")
    print(f"  Interface: {MLFLOW_TRACKING_URI} (abra no browser)")
    print()
    print("O que foi salvo:")
    print(f"  Parâmetros: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}")
    print(f"  Métricas:   accuracy={acc:.4f}, f1_score={f1:.4f}")
    print()
    print("Dica: rode este script novamente mudando n_estimators ou max_depth")
    print("e compare os resultados na interface do MLflow em localhost:5000.")

print()
print("=" * 60)
print("RESUMO FINAL")
print("=" * 60)
print(f"  Dataset:    Iris ({len(X)} flores, {X.shape[1]} features, 3 espécies)")
print(f"  Modelo:     RandomForestClassifier ({params['n_estimators']} árvores)")
print(f"  Accuracy:   {acc*100:.2f}%")
print(f"  F1 Score:   {f1:.4f}")
print(f"  MLflow:     experimento salvo em {MLFLOW_TRACKING_URI}")
print("=" * 60)
