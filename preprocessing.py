import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# ETAPA 3 DO CRISP-DM: LIMPEZA E PREPARAÇÃO DOS DADOS (Data Preparation)
#
# Suas anotações resumem bem:
# "Agora que você sabe os problemas [via EDA], precisa corrigi-los."
#
# O que faremos aqui:
#   3.1 — Imputação: preencher dados faltantes
#   3.2 — Remoção de outliers severos
#   3.3 — One-Hot Encoding: converter texto em número
#   3.4 — Normalização / Escalonamento
#
# IMPORTANTE: O Iris original é um dataset limpo. Para o aprendizado fazer
# sentido, vamos injetar problemas propositalmente e depois corrigi-los.
# Isso é o que você encontrará em dados reais.
# ==============================================================================

# --- Carregar e montar o dataset "sujo" (simulando dados do mundo real) -------
iris = load_iris()
df = pd.DataFrame(iris.data, columns=["sepal_len", "sepal_wid", "petal_len", "petal_wid"])
df["target"] = iris.target

np.random.seed(42)

# Injetando coluna categórica (como sex/cp no dataset de doenças cardíacas)
df["regiao"] = np.random.choice(["norte", "sul", "leste"], size=len(df))

# Injetando valores faltantes (15% das linhas em duas colunas)
idx_nulos_1 = np.random.choice(df.index, size=22, replace=False)
idx_nulos_2 = np.random.choice(df.index, size=18, replace=False)
df.loc[idx_nulos_1, "sepal_len"] = np.nan
df.loc[idx_nulos_2, "petal_wid"] = np.nan

# Injetando outliers severos (erros de medição)
df.loc[7,  "sepal_wid"] = 15.0   # pressão arterial 280 — claramente errado
df.loc[42, "petal_len"] = -3.5   # valor negativo impossível para comprimento
df.loc[99, "sepal_len"] = 99.0   # impossível para essa feature

print("=" * 60)
print("DATASET ORIGINAL (com problemas injetados)")
print("=" * 60)
print(f"\nDimensões: {df.shape[0]} linhas × {df.shape[1]} colunas")
print(f"Colunas:   {list(df.columns)}")
print(f"\nPrimeiras 5 linhas (dados brutos):")
print(df.head().to_string())

nulos_total = df.isnull().sum().sum()
print(f"\nProblemas detectados:")
print(f"  Valores nulos:   {nulos_total}")
print(f"  Outliers severos injetados: 3 (linhas 7, 42, 99)")
print(f"  Coluna categórica: 'regiao' (norte/sul/leste) → texto, não número")


# ==============================================================================
# ETAPA 3.1 — IMPUTAÇÃO: preencher valores faltantes
#
# Regra das suas anotações:
#   Colunas numéricas  → mediana (não a média!)
#   Colunas categóricas → moda (valor mais frequente)
#
# Por que mediana e não média?
# A média é puxada por outliers. Se uma coluna tem um valor de 99.0
# (injetado acima), a média vai ser distorcida. A mediana (valor do meio)
# ignora os extremos e representa melhor o "típico".
# ==============================================================================
print()
print("=" * 60)
print("ETAPA 3.1 — IMPUTAÇÃO (preenchimento de nulos)")
print("=" * 60)

df_limpo = df.copy()

colunas_numericas = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"]
colunas_categoricas = ["regiao"]

print(f"\n--- Imputação de colunas NUMÉRICAS com a MEDIANA ---")
print()
for col in colunas_numericas:
    nulos_antes = df_limpo[col].isnull().sum()
    if nulos_antes == 0:
        print(f"  {col:<12} → sem nulos, nada a fazer")
        continue

    media   = df_limpo[col].mean()
    mediana = df_limpo[col].median()

    df_limpo[col] = df_limpo[col].fillna(mediana)
    nulos_depois = df_limpo[col].isnull().sum()

    print(f"  {col:<12} → {nulos_antes} nulos encontrados")
    print(f"    média   = {media:.4f}  (poderia distorcer se tiver outliers)")
    print(f"    mediana = {mediana:.4f}  ← usamos esta")
    print(f"    resultado: {nulos_antes} nulos → {nulos_depois} nulos")
    print()

print(f"--- Imputação de colunas CATEGÓRICAS com a MODA ---")
print()
for col in colunas_categoricas:
    nulos_antes = df_limpo[col].isnull().sum()
    moda = df_limpo[col].mode()[0]
    contagem = df_limpo[col].value_counts()

    print(f"  {col} → distribuição atual:")
    for valor, qtd in contagem.items():
        marcador = " ← moda (mais frequente)" if valor == moda else ""
        print(f"    {valor:<10} {qtd} ocorrências{marcador}")

    if nulos_antes > 0:
        df_limpo[col] = df_limpo[col].fillna(moda)
        print(f"  {nulos_antes} nulos preenchidos com '{moda}'")
    else:
        print(f"  Sem nulos. Moda = '{moda}' (informação para referência)")
    print()

nulos_restantes = df_limpo.isnull().sum().sum()
print(f"  Nulos restantes no dataset: {nulos_restantes} ✓")


# ==============================================================================
# ETAPA 3.2 — REMOÇÃO DE OUTLIERS SEVEROS
#
# Outliers moderados → geralmente mantemos (podem ser dados legítimos)
# Outliers SEVEROS   → removemos se forem erros claros de medição
#
# Usamos o método IQR para definir limites aceitáveis.
# Valores além de 3×IQR do quartil são considerados erros.
# ==============================================================================
print()
print("=" * 60)
print("ETAPA 3.2 — REMOÇÃO DE OUTLIERS SEVEROS")
print("=" * 60)
print()
print("Estratégia: IQR com fator 3.0 (apenas erros claros de medição)")
print("  Limite inferior = Q1 - 3.0 × IQR")
print("  Limite superior = Q3 + 3.0 × IQR")
print()

linhas_antes = len(df_limpo)
mascara_valido = pd.Series([True] * len(df_limpo), index=df_limpo.index)

for col in colunas_numericas:
    Q1  = df_limpo[col].quantile(0.25)
    Q3  = df_limpo[col].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf = Q1 - 3.0 * IQR
    lim_sup = Q3 + 3.0 * IQR

    outliers = df_limpo[(df_limpo[col] < lim_inf) | (df_limpo[col] > lim_sup)]
    col_valido = (df_limpo[col] >= lim_inf) & (df_limpo[col] <= lim_sup)
    mascara_valido &= col_valido

    status = f"⚠  {len(outliers)} outlier(s) severo(s)" if len(outliers) > 0 else "✓ ok"
    print(f"  {col:<12}  faixa [{lim_inf:.2f}, {lim_sup:.2f}]  →  {status}")
    for idx, row in outliers.iterrows():
        print(f"    linha {idx}: valor = {row[col]:.2f} (fora da faixa)")

df_limpo = df_limpo[mascara_valido].reset_index(drop=True)
linhas_depois = len(df_limpo)

print()
print(f"  Linhas antes:  {linhas_antes}")
print(f"  Linhas depois: {linhas_depois}")
print(f"  Removidas:     {linhas_antes - linhas_depois}")
print()
print("  Atenção: não remova outliers moderados sem investigar!")
print("  Um valor incomum pode ser exatamente o caso raro que o")
print("  modelo precisa aprender a identificar.")


# ==============================================================================
# ETAPA 3.3 — ONE-HOT ENCODING: converter texto em número
#
# Algoritmos de ML trabalham APENAS com números.
# A coluna 'regiao' tem texto: "norte", "sul", "leste".
#
# Não podemos usar label encoding (0, 1, 2) porque isso implica uma
# ordem: o modelo pensaria que leste (2) > sul (1) > norte (0).
# Mas regiões não têm ordem!
#
# One-Hot Encoding resolve: cria uma coluna binária (0/1) para cada
# categoria. A flor ou está na região ou não está.
#
# Antes:  regiao = "norte"
# Depois: regiao_norte=1, regiao_sul=0, regiao_leste=0
#
# Analogia das suas anotações:
# "É como converter um enum de string para flags booleanas no seu código."
# ==============================================================================
print()
print("=" * 60)
print("ETAPA 3.3 — ONE-HOT ENCODING (texto → números)")
print("=" * 60)

print(f"\n--- Antes do encoding ---")
print(f"  Coluna 'regiao' com valores únicos: {sorted(df_limpo['regiao'].unique())}")
print(f"  Distribuição:")
for val, qtd in df_limpo["regiao"].value_counts().items():
    print(f"    {val:<10} {qtd} amostras")

print(f"\n--- Aplicando pd.get_dummies() ---")
print(f"  drop_first=True remove uma coluna redundante.")
print(f"  (se regiao_norte=0 e regiao_sul=0, sabemos que é leste)")
print(f"  Isso evita o problema de 'dummy variable trap' (multicolinearidade)")

df_encoded = pd.get_dummies(df_limpo, columns=["regiao"], drop_first=True, dtype=int)

print(f"\n--- Depois do encoding ---")
novas_colunas = [c for c in df_encoded.columns if c.startswith("regiao_")]
print(f"  Coluna 'regiao' virou: {novas_colunas}")
print()

exemplo = df_limpo[["regiao"]].head(6).copy()
exemplo["regiao_leste"] = df_encoded["regiao_leste"].head(6).values
exemplo["regiao_sul"]   = df_encoded["regiao_sul"].head(6).values
print(f"  {'regiao':<12} {'regiao_leste':>14} {'regiao_sul':>12}  (norte = ambos 0)")
print(f"  {'-'*42}")
for _, row in exemplo.iterrows():
    print(f"  {row['regiao']:<12} {row['regiao_leste']:>14} {row['regiao_sul']:>12}")


# ==============================================================================
# ETAPA 3.4 — NORMALIZAÇÃO / ESCALONAMENTO (StandardScaler)
#
# Problema: features em escalas muito diferentes confundem alguns modelos.
#
# Exemplo com o Iris:
#   sepal_len: varia de 4.3 a 7.9  (escala ~8)
#   petal_wid: varia de 0.1 a 2.5  (escala ~2)
#
# Para algoritmos baseados em distância (KNN, SVM, redes neurais),
# uma feature com valores maiores domina o cálculo injustamente.
# Random Forest não precisa, mas é boa prática padronizar.
#
# StandardScaler: transforma cada feature para média=0, desvio padrão=1.
#   valor_normalizado = (valor - média) / desvio_padrão
# ==============================================================================
print()
print("=" * 60)
print("ETAPA 3.4 — NORMALIZAÇÃO (StandardScaler)")
print("=" * 60)

print(f"\n--- Antes da normalização ---")
print(f"  {'Feature':<14} {'min':>8} {'max':>8} {'média':>8} {'std':>8}")
print(f"  {'-'*50}")
for col in colunas_numericas:
    print(f"  {col:<14} {df_encoded[col].min():>8.2f} {df_encoded[col].max():>8.2f} "
          f"{df_encoded[col].mean():>8.2f} {df_encoded[col].std():>8.2f}")

scaler = StandardScaler()
df_scaled = df_encoded.copy()
df_scaled[colunas_numericas] = scaler.fit_transform(df_encoded[colunas_numericas])

print(f"\n--- Depois da normalização ---")
print(f"  {'Feature':<14} {'min':>8} {'max':>8} {'média':>8} {'std':>8}")
print(f"  {'-'*50}")
for col in colunas_numericas:
    print(f"  {col:<14} {df_scaled[col].min():>8.2f} {df_scaled[col].max():>8.2f} "
          f"{df_scaled[col].mean():>8.2f} {df_scaled[col].std():>8.2f}")

print()
print("  Todas as features agora têm média ≈ 0 e desvio padrão ≈ 1.")
print("  Nenhuma feature vai 'dominar' o modelo só por ter valores maiores.")
print()
print("  Quando normalizar:")
print("    SIM → KNN, SVM, Regressão Logística, Redes Neurais")
print("    NÃO é necessário → Random Forest, Decision Tree, XGBoost")
print("    (árvores de decisão não usam distâncias, são imunes à escala)")


# ==============================================================================
# RESULTADO FINAL: dataset pronto para o modelo
# ==============================================================================
print()
print("=" * 60)
print("RESULTADO: DATASET LIMPO E PRONTO PARA O MODELO")
print("=" * 60)

X = df_scaled.drop(columns=["target"])
y = df_scaled["target"]

print(f"\n  Linhas originais:     150")
print(f"  Linhas após limpeza: {len(df_scaled)} (removidas: {150 - len(df_scaled)} outliers severos)")
print(f"\n  Features de entrada (X): {list(X.columns)}")
print(f"  Feature alvo       (y):  target (0=setosa, 1=versicolor, 2=virginica)")
print(f"\n  Shape final:  X={X.shape}  y={y.shape}")

print(f"\n  Primeiras 3 linhas do dataset processado:")
print(f"  {X.head(3).to_string()}")

print()
print("=" * 60)
print("RESUMO DO PIPELINE DE PREPARAÇÃO")
print("=" * 60)
print()
print("  [3.1] Imputação")
print("          → numérico:    fillna(mediana)")
print("          → categórico:  fillna(moda)")
print("  [3.2] Outliers severos removidos via IQR × 3.0")
print("  [3.3] One-Hot Encoding em 'regiao' (texto → 0/1)")
print("          drop_first=True evita multicolinearidade")
print("  [3.4] StandardScaler nas features numéricas")
print("          média=0, std=1")
print()
print("  Próximo passo: passar X e y para o train.py e treinar o modelo!")
print("=" * 60)
