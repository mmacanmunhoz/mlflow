import numpy as np
import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# AULA 07 — SELEÇÃO DE MODELOS E VALIDAÇÃO CRUZADA
#
# Suas anotações definem bem o problema central:
# "Um modelo pode ter ótimo resultado no treino, mas falhar com dados novos."
#
# Dois extremos que queremos evitar:
#   UNDERFITTING → modelo simples demais, não aprende o padrão
#   OVERFITTING  → modelo decorou o treino, não generaliza
#
# Objetivo: encontrar o ponto de equilíbrio.
# ==============================================================================

iris = load_iris()
X, y = iris.data, iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ==============================================================================
# PARTE 1: DEMONSTRANDO OVERFITTING E UNDERFITTING
#
# A melhor forma de entender é VER os números.
# Vamos usar uma Árvore de Decisão com profundidades diferentes:
#
#   max_depth=1 → árvore rasa = underfitting (simples demais)
#   max_depth=30 → árvore profunda sem limite = overfitting (complexa demais)
#   max_depth=3  → ponto de equilíbrio
# ==============================================================================
print("=" * 60)
print("PARTE 1: OVERFITTING vs UNDERFITTING na prática")
print("=" * 60)
print()
print("Usando DecisionTreeClassifier com diferentes max_depth:")
print()
print(f"  {'max_depth':<12} {'Treino':>10} {'Teste':>10} {'Diferença':>12}  {'Diagnóstico'}")
print(f"  {'-' * 65}")

profundidades = [1, 2, 3, 5, 8, 15, 30]
for depth in profundidades:
    modelo = DecisionTreeClassifier(max_depth=depth, random_state=42)
    modelo.fit(X_train, y_train)

    acc_treino = accuracy_score(y_train, modelo.predict(X_train))
    acc_teste  = accuracy_score(y_test,  modelo.predict(X_test))
    diferenca  = acc_treino - acc_teste

    if diferenca > 0.10:
        diagnostico = "OVERFITTING ← modelo decorou o treino"
    elif acc_treino < 0.80:
        diagnostico = "UNDERFITTING ← modelo simples demais"
    else:
        diagnostico = "bom equilíbrio ✓"

    print(f"  {str(depth):<12} {acc_treino:>10.4f} {acc_teste:>10.4f} {diferenca:>+12.4f}  {diagnostico}")

print()
print("Repare:")
print("  depth=1  → treino baixo E teste baixo = underfitting")
print("  depth=30 → treino=1.0 MAS teste cai   = overfitting")
print("  depth=3  → treino e teste próximos     = ponto ideal")
print()
print("Analogia SRE: é como um threshold de alerta.")
print("  Muito alto → deixa passar incidentes (underfitting)")
print("  Muito baixo → dispara o tempo todo, falso positivo (overfitting)")


# ==============================================================================
# PARTE 2: K-FOLD CROSS VALIDATION
#
# Suas anotações: "Uma única divisão treino/teste pode dar sorte ou azar."
#
# Com K-Fold, dividimos os dados em K partes (folds).
# O modelo é treinado K vezes, cada vez usando um fold diferente como
# validação e os demais como treino.
#
# Resultado: K scores → tiramos a média e o desvio padrão.
# Isso dá uma estimativa muito mais confiável do desempenho real.
# ==============================================================================
print()
print("=" * 60)
print("PARTE 2: K-FOLD CROSS VALIDATION")
print("=" * 60)
print()
print("Problema com uma única divisão treino/teste:")
print("  Se os 20% de teste forem fáceis por acaso → score inflado")
print("  Se forem difíceis → score deflacionado")
print("  Não sabemos qual dos dois aconteceu!")
print()
print("K-Fold resolve: roda K vezes com grupos diferentes, tira a média.")
print()

modelo_kfold = RandomForestClassifier(n_estimators=100, random_state=42)

for k in [3, 5, 10]:
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    # StratifiedKFold garante que cada fold mantém a proporção das classes
    # Suas anotações: "mantém proporção das classes em cada divisão"

    scores = cross_val_score(modelo_kfold, X, y, cv=kf, scoring="accuracy")

    print(f"  K={k} (StratifiedKFold):")
    print(f"    Scores por fold: {[f'{s:.4f}' for s in scores]}")
    print(f"    Média:           {scores.mean():.4f}")
    print(f"    Desvio padrão:   {scores.std():.4f}  ← quanto o score varia entre folds")
    print(f"    Intervalo:       [{scores.mean() - scores.std():.4f}, {scores.mean() + scores.std():.4f}]")
    print()

print("Interpretando o desvio padrão:")
print("  std baixo → modelo consistente entre diferentes subsets dos dados")
print("  std alto  → modelo instável, performance depende muito de quais")
print("              dados foram para treino (sinal de problema)")
print()
print("Por que StratifiedKFold e não KFold simples?")
print("  KFold simples pode colocar todas as flores 'setosa' no mesmo fold.")
print("  StratifiedKFold garante que cada fold tem ~33% de cada espécie.")
print("  Suas anotações: 'mantém proporção das classes em cada divisão'")


# ==============================================================================
# PARTE 3: COMPARANDO MODELOS DIFERENTES
#
# Suas anotações citam a ordem de complexidade crescente:
#   Logistic Regression (mais simples, MVP)
#   Decision Tree
#   Random Forest (mais complexo)
#
# Usamos cross-validation para comparar todos de forma justa.
# ==============================================================================
print()
print("=" * 60)
print("PARTE 3: COMPARANDO MODELOS COM CROSS-VALIDATION")
print("=" * 60)
print()
print("Usando StratifiedKFold com K=5 para comparar 3 modelos:")
print()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

candidatos = {
    "LogisticRegression (MVP)": LogisticRegression(max_iter=200, random_state=42),
    "DecisionTree        ": DecisionTreeClassifier(max_depth=3, random_state=42),
    "RandomForest        ": RandomForestClassifier(n_estimators=100, random_state=42),
}

resultados = {}
print(f"  {'Modelo':<28} {'Média':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print(f"  {'-' * 60}")

for nome, modelo in candidatos.items():
    scores = cross_val_score(modelo, X, y, cv=cv, scoring="accuracy")
    resultados[nome] = scores
    print(f"  {nome} {scores.mean():>8.4f} {scores.std():>8.4f} {scores.min():>8.4f} {scores.max():>8.4f}")

melhor_nome = max(resultados, key=lambda k: resultados[k].mean())
print()
print(f"  Melhor modelo por média: {melhor_nome.strip()}")
print()
print("  Logistic Regression é o MVP — simples, rápido e explicável.")
print("  Se já performa bem, não há motivo para complexidade maior.")
print("  Suas anotações: 'crie o modelo mais simples possível primeiro'")


# ==============================================================================
# PARTE 4: GRID SEARCH — buscando os melhores hiperparâmetros
#
# Suas anotações:
#   "Grid Search → testa várias combinações possíveis"
#   "Random Search → testa combinações aleatórias"
#
# GridSearchCV combina Grid Search + Cross Validation automaticamente.
# Para cada combinação de hiperparâmetros, ele roda K-Fold e retorna
# a combinação com melhor performance média.
# ==============================================================================
print()
print("=" * 60)
print("PARTE 4: GRID SEARCH (busca de hiperparâmetros)")
print("=" * 60)
print()
print("Hiperparâmetros = configurações do modelo que VOCÊ define antes do treino.")
print("Exemplo: quantas árvores? Quão profundas podem ser?")
print()
print("Grid Search testa todas as combinações e usa cross-validation")
print("para avaliar cada uma. Retorna a melhor configuração.")
print()

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth":    [3, 5, 10, None],
    "min_samples_split": [2, 5],
}

total_combinacoes = (
    len(param_grid["n_estimators"])
    * len(param_grid["max_depth"])
    * len(param_grid["min_samples_split"])
)

print(f"  Grade de busca:")
for param, valores in param_grid.items():
    print(f"    {param}: {valores}")
print(f"  Total de combinações: {total_combinacoes}")
print(f"  Com cv=5: {total_combinacoes * 5} treinamentos ao todo")
print()
print("  Rodando Grid Search... (pode demorar alguns segundos)")

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,    # usa todos os núcleos da CPU em paralelo
    verbose=0,
)
grid_search.fit(X_train, y_train)

print()
print(f"  Melhor combinação encontrada:")
for param, valor in grid_search.best_params_.items():
    print(f"    {param}: {valor}")
print(f"  Score médio (cross-val): {grid_search.best_score_:.4f}")

melhor_modelo = grid_search.best_estimator_
acc_final = accuracy_score(y_test, melhor_modelo.predict(X_test))
print(f"  Accuracy no conjunto de teste: {acc_final:.4f}")

print()
print("  Top 5 combinações testadas:")
print(f"  {'rank':>5} {'n_est':>6} {'depth':>7} {'min_split':>10} {'score_cv':>10}")
print(f"  {'-' * 45}")

resultados_grid = grid_search.cv_results_
indices = np.argsort(resultados_grid["rank_test_score"])[:5]
for i in indices:
    rank      = resultados_grid["rank_test_score"][i]
    n_est     = resultados_grid["param_n_estimators"][i]
    depth     = resultados_grid["param_max_depth"][i]
    min_split = resultados_grid["param_min_samples_split"][i]
    score     = resultados_grid["mean_test_score"][i]
    print(f"  {rank:>5} {str(n_est):>6} {str(depth):>7} {str(min_split):>10} {score:>10.4f}")


# ==============================================================================
# PARTE 5: RANDOM SEARCH — quando o grid é grande demais
#
# Grid Search com muitos hiperparâmetros explode em combinações.
# 5 parâmetros com 5 valores cada = 3125 combinações × 5 folds = 15625 treinos.
#
# Random Search resolve: em vez de testar tudo, testa N combinações aleatórias.
# Na prática, encontra boas configurações com muito menos treinos.
# ==============================================================================
print()
print("=" * 60)
print("PARTE 5: RANDOM SEARCH (busca aleatória)")
print("=" * 60)
print()
print("Quando o Grid Search é muito lento (muitos parâmetros),")
print("Random Search testa combinações aleatórias.")
print("Pesquisas mostram que ele encontra bons resultados com ~20%")
print("das tentativas do Grid Search.")
print()

param_dist = {
    "n_estimators":      [50, 100, 150, 200, 300],
    "max_depth":         [3, 5, 7, 10, 15, None],
    "min_samples_split": [2, 4, 6, 8, 10],
    "min_samples_leaf":  [1, 2, 3, 4],
    "max_features":      ["sqrt", "log2", None],
}

total_possivel = (
    len(param_dist["n_estimators"])
    * len(param_dist["max_depth"])
    * len(param_dist["min_samples_split"])
    * len(param_dist["min_samples_leaf"])
    * len(param_dist["max_features"])
)

n_iter = 20

print(f"  Espaço total de combinações possíveis: {total_possivel}")
print(f"  Random Search vai testar apenas: {n_iter} ({n_iter/total_possivel*100:.1f}%)")
print(f"  Com cv=5: {n_iter * 5} treinamentos ao invés de {total_possivel * 5}")
print()
print("  Rodando Random Search...")

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=n_iter,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
    verbose=0,
)
random_search.fit(X_train, y_train)

print()
print(f"  Melhor combinação encontrada:")
for param, valor in random_search.best_params_.items():
    print(f"    {param}: {valor}")
print(f"  Score médio (cross-val): {random_search.best_score_:.4f}")

acc_random = accuracy_score(y_test, random_search.best_estimator_.predict(X_test))
print(f"  Accuracy no conjunto de teste: {acc_random:.4f}")

print()
print("  Quando usar cada um:")
print("    Grid Search  → grid pequeno (<1000 combinações), precisa ser exaustivo")
print("    Random Search → grid grande, quer resultado rápido")
print("    Bayesiano    → produção, quando otimização fina importa muito")


# ==============================================================================
# PARTE 6: REGISTRANDO A COMPARAÇÃO NO MLFLOW
#
# Cada modelo testado com suas métricas de cross-validation fica registrado.
# Depois você abre localhost:5000 e compara todos os experimentos visualmente.
# ==============================================================================
print()
print("=" * 60)
print("PARTE 6: REGISTRANDO EXPERIMENTOS NO MLFLOW")
print("=" * 60)

MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("iris-model-selection")

cv_final = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

modelos_para_registrar = {
    "logistic_regression": LogisticRegression(max_iter=200, random_state=42),
    "decision_tree_depth3": DecisionTreeClassifier(max_depth=3, random_state=42),
    "random_forest_default": RandomForestClassifier(n_estimators=100, random_state=42),
    "random_forest_grid_best": grid_search.best_estimator_,
}

print()
for nome, modelo in modelos_para_registrar.items():
    scores_cv = cross_val_score(modelo, X_train, y_train, cv=cv_final, scoring="accuracy")
    modelo.fit(X_train, y_train)
    acc_teste_final = accuracy_score(y_test, modelo.predict(X_test))
    f1_teste_final  = f1_score(y_test, modelo.predict(X_test), average="weighted")

    with mlflow.start_run(run_name=nome):
        mlflow.log_metric("cv_accuracy_mean", scores_cv.mean())
        mlflow.log_metric("cv_accuracy_std",  scores_cv.std())
        mlflow.log_metric("test_accuracy",    acc_teste_final)
        mlflow.log_metric("test_f1",          f1_teste_final)

        run_id = mlflow.active_run().info.run_id
        print(f"  [{nome}]")
        print(f"    CV accuracy: {scores_cv.mean():.4f} ± {scores_cv.std():.4f}")
        print(f"    Test accuracy: {acc_teste_final:.4f}")
        print(f"    Run ID: {run_id[:8]}...")
        print()

print(f"  Abra {MLFLOW_TRACKING_URI} para comparar todos os modelos!")
print(f"  Experimento: 'iris-model-selection'")

print()
print("=" * 60)
print("RESUMO FINAL")
print("=" * 60)
print()
print("  OVERFITTING:  treino muito > teste → modelo decorou os dados")
print("  UNDERFITTING: treino baixo E teste baixo → modelo simples demais")
print()
print("  K-FOLD CV:    divide em K partes, treina K vezes, tira a média")
print("                → estimativa mais confiável do desempenho real")
print()
print("  STRATIFIED:   garante proporção das classes em cada fold")
print("                → essencial para datasets desbalanceados")
print()
print("  GRID SEARCH:  testa TODAS as combinações de hiperparâmetros")
print("  RANDOM SEARCH: testa N combinações aleatórias — mais rápido")
print()
print("  REGRA DE OURO: comece simples (Logistic Regression), meça,")
print("  depois adicione complexidade somente se necessário.")
print("=" * 60)
