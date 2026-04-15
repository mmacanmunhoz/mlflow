import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from scipy import stats

# ==============================================================================
# EDA — Análise Exploratória de Dados (Exploratory Data Analysis)
#
# Suas anotações definem bem:
# "EDA = você investigando os dados antes de construir qualquer coisa."
# "Pense como um troubleshooting: antes de corrigir um problema, você lê
#  os logs, olha as métricas, entende o que está acontecendo."
#
# Os 4 pilares do EDA (conforme suas anotações):
#   1. Visão geral dos dados (shape, tipos, .info(), .describe())
#   2. Dados Faltantes (Missing Values)
#   3. Distribuição da variável alvo (Target)
#   4. Outliers
#   + Bônus: Correlações
# ==============================================================================

# Carregando os dados e convertendo para DataFrame do pandas
# DataFrame = tabela estruturada, igual a uma planilha do Excel — mas programável
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
df["especie"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

# Nomes curtos para facilitar a leitura no console
df.columns = ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "target", "especie"]

print("=" * 60)
print("EDA — PILAR 1: VISÃO GERAL DOS DADOS")
print("=" * 60)

# ==============================================================================
# .shape → retorna (linhas, colunas). Sempre a primeira coisa a olhar.
# ==============================================================================
print(f"\nDimensões do dataset: {df.shape[0]} linhas × {df.shape[1]} colunas")
print("  → cada linha é uma flor, cada coluna é uma informação sobre ela\n")

# ==============================================================================
# .info() → mostra tipos de dados e contagem de valores não-nulos.
# Analogia das suas anotações: é como rodar `df -h` ou `top` num servidor novo.
# ==============================================================================
print("--- df.info() ---")
print(f"{'Coluna':<15} {'Tipo':<12} {'Não-nulos':<12} {'Exemplo'}")
print("-" * 55)
for col in df.columns:
    tipo = str(df[col].dtype)
    nao_nulos = df[col].notna().sum()
    exemplo = str(df[col].iloc[0])
    print(f"{col:<15} {tipo:<12} {nao_nulos:<12} {exemplo}")

# ==============================================================================
# .describe() → estatísticas descritivas: média, desvio padrão, min, max, quartis.
# Dá uma "radiografia" de cada coluna numérica.
# ==============================================================================
print(f"\n--- df.describe() ---")
numeric_cols = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"]
stats_labels = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
desc = df[numeric_cols].describe()

print(f"{'':>8}", end="")
for col in numeric_cols:
    print(f"  {col:>10}", end="")
print()
print("-" * 52)
for stat in stats_labels:
    print(f"{stat:>8}", end="")
    for col in numeric_cols:
        print(f"  {desc.loc[stat, col]:>10.2f}", end="")
    print()

print()
print("O que procurar aqui:")
print("  mean  → valor médio. Se muito diferente da mediana (50%), pode ter outliers.")
print("  std   → desvio padrão. Alto = dados muito espalhados.")
print("  min/max → se esses valores parecem absurdos, pode ser erro nos dados.")

print()
print("=" * 60)
print("EDA — PILAR 2: DADOS FALTANTES (Missing Values)")
print("=" * 60)

# ==============================================================================
# Dados faltantes = campos vazios (NaN = Not a Number).
# Algoritmos de ML NÃO conseguem trabalhar com NaN.
# Você precisa decidir: remover a linha, preencher com uma estimativa, ou ignorar.
#
# Analogia das suas anotações:
# "É como ter um serviço que não manda todas as métricas pro seu sistema
#  de monitoramento. Você precisa decidir o que fazer com esse buraco."
# ==============================================================================
print(f"\n--- Contagem de NaN por coluna (df.isnull().sum()) ---")
print()

nulos = df.isnull().sum()
total = len(df)

for col in df.columns:
    qtd = nulos[col]
    pct = (qtd / total) * 100
    barra = "█" * int(pct / 2) if qtd > 0 else "✓ sem nulos"
    print(f"  {col:<15} {qtd:>4} nulos ({pct:>5.1f}%)  {barra}")

print()
print("Iris é um dataset limpo → 0 nulos em todas as colunas.")
print("Na prática (ex: dataset médico das suas anotações), é comum ter 20-50%")
print("de dados faltando. Estratégias de preenchimento (imputação):")
print()
print("  Colunas numéricas  → preenche com a MEDIANA (não a média!)")
print("    Por quê mediana? Se um colesterol é 600 (outlier), a média sobe")
print("    muito e distorce o preenchimento. A mediana é mais robusta.")
print()
print("  Colunas categóricas → preenche com a MODA (valor mais frequente)")
print()

# Simulando dados faltantes para demonstrar como tratar
print("--- Simulação: e se tivéssemos nulos? ---")
df_com_nulos = df.copy()
np.random.seed(42)
indices_nulos = np.random.choice(df_com_nulos.index, size=15, replace=False)
df_com_nulos.loc[indices_nulos, "sepal_len"] = np.nan

nulos_simulados = df_com_nulos["sepal_len"].isnull().sum()
mediana = df_com_nulos["sepal_len"].median()
media = df_com_nulos["sepal_len"].mean()

print(f"  sepal_len com {nulos_simulados} nulos injetados artificialmente")
print(f"  Média da coluna:   {media:.4f}  ← pode ser distorcida por outliers")
print(f"  Mediana da coluna: {mediana:.4f}  ← mais segura para imputação")

df_com_nulos["sepal_len"] = df_com_nulos["sepal_len"].fillna(mediana)
print(f"  Após fillna(mediana): {df_com_nulos['sepal_len'].isnull().sum()} nulos restantes")

print()
print("=" * 60)
print("EDA — PILAR 3: DISTRIBUIÇÃO DA VARIÁVEL ALVO (Target)")
print("=" * 60)

# ==============================================================================
# Target = a coluna que o modelo vai tentar prever.
# Precisamos verificar se as classes estão equilibradas.
#
# Exemplo das suas anotações:
# "Se 90% dos pacientes fossem saudáveis, um modelo burro que sempre dissesse
#  'saudável' acertaria 90% das vezes. Mas não valeria nada."
#
# Isso se chama DATASET DESBALANCEADO e é um problema sério em ML.
# ==============================================================================
print(f"\n--- Distribuição das classes ---")
print()

contagem = df["especie"].value_counts()
total = len(df)

for especie, qtd in contagem.items():
    pct = (qtd / total) * 100
    barra = "█" * int(pct / 2)
    print(f"  {especie:<12} {qtd:>4} amostras ({pct:>5.1f}%)  {barra}")

print()
maior = contagem.max()
menor = contagem.min()
razao = maior / menor
print(f"  Razão maior/menor classe: {razao:.1f}x")

if razao < 1.5:
    print("  → Dataset BALANCEADO ✓ (razão < 1.5)")
    print("     A acurácia é uma boa métrica aqui.")
elif razao < 3:
    print("  → Dataset LEVEMENTE DESBALANCEADO ⚠")
    print("     Use F1 Score além da acurácia.")
else:
    print("  → Dataset MUITO DESBALANCEADO ✗")
    print("     Acurácia engana! Use F1, Precision, Recall.")
    print("     Considere técnicas: SMOTE, class_weight='balanced'.")

print()
print("Iris é balanceado (50 amostras por classe) → raridade no mundo real.")
print("Na maioria dos problemas reais o desbalanceamento é comum:")
print("  Fraude bancária: 99% transações legítimas, 1% fraude")
print("  Falha de máquina: 95% operação normal, 5% falha")

print()
print("=" * 60)
print("EDA — PILAR 4: OUTLIERS (Valores Absurdos)")
print("=" * 60)

# ==============================================================================
# Outlier = valor que destoa completamente dos outros.
# Pode ser um erro de medição, erro de digitação, ou um caso legítimo raro.
#
# Analogia das suas anotações:
# "É como um spike estranho num gráfico de latência. Pode ser um problema
#  real, pode ser ruído. Você precisa investigar antes de agir."
#
# Duas formas de detectar:
#   1. IQR (Interquartile Range) → método estatístico robusto
#   2. Z-Score → mede distância da média em desvios padrão (>3 = outlier)
# ==============================================================================

print("\n--- Método 1: IQR (Interquartile Range) ---")
print()
print("  IQR = distância entre o 25º e 75º percentil (a 'faixa normal')")
print("  Outlier = qualquer valor abaixo de Q1 - 1.5×IQR")
print("            ou acima de Q3 + 1.5×IQR")
print()

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    limite_inf = Q1 - 1.5 * IQR
    limite_sup = Q3 + 1.5 * IQR
    outliers = df[(df[col] < limite_inf) | (df[col] > limite_sup)]
    qtd = len(outliers)

    status = "⚠ investigar" if qtd > 0 else "✓ ok"
    print(f"  {col:<12} Q1={Q1:.2f}  Q3={Q3:.2f}  IQR={IQR:.2f}")
    print(f"    faixa normal: [{limite_inf:.2f}, {limite_sup:.2f}]")
    print(f"    outliers encontrados: {qtd}  {status}")
    if qtd > 0:
        valores = df[col][outliers.index].values
        print(f"    valores: {valores[:5]}")
    print()

print("--- Método 2: Z-Score ---")
print()
print("  Z-Score = (valor - média) / desvio_padrão")
print("  Mede quantos desvios padrão o valor está da média.")
print("  Regra geral: |Z| > 3 → provável outlier")
print()

for col in numeric_cols:
    z_scores = np.abs(stats.zscore(df[col]))
    outliers_z = df[z_scores > 3]
    qtd = len(outliers_z)
    max_z = z_scores.max()
    status = "⚠ investigar" if qtd > 0 else "✓ ok"
    print(f"  {col:<12} Z-Score máximo: {max_z:.2f}   outliers (|Z|>3): {qtd}  {status}")

print()
print("Iris tem poucos/nenhum outlier severo — dados bem coletados.")
print("O que fazer quando encontrar outliers:")
print("  1. Investigar se é erro de dado (corrigir ou remover)")
print("  2. Se for dado legítimo: deixar (pode ser informação importante)")
print("  3. Usar modelos robustos a outliers (ex: árvores de decisão)")

print()
print("=" * 60)
print("EDA — BÔNUS: CORRELAÇÕES ENTRE FEATURES")
print("=" * 60)

# ==============================================================================
# Correlação = duas colunas sobem e descem juntas?
# Valor entre -1 e 1:
#   +1 = correlação positiva perfeita (A sobe, B sobe junto)
#    0 = sem correlação
#   -1 = correlação negativa perfeita (A sobe, B desce)
#
# Por que isso importa?
# - Features muito correlacionadas entre si = informação redundante
# - Features correlacionadas com o target = boas candidatas preditoras
# ==============================================================================
print()
corr = df[numeric_cols].corr()

print("Matriz de correlação (valores entre -1 e 1):")
print()
print(f"  {'':>12}", end="")
for col in numeric_cols:
    print(f"  {col[:8]:>10}", end="")
print()
print("  " + "-" * 52)

for col1 in numeric_cols:
    print(f"  {col1[:12]:>12}", end="")
    for col2 in numeric_cols:
        val = corr.loc[col1, col2]
        if col1 == col2:
            print(f"  {'----':>10}", end="")
        else:
            destaque = " ◄" if abs(val) > 0.7 and col1 != col2 else ""
            print(f"  {val:>+9.2f}{destaque}", end="")
    print()

print()
print("Interpretando:")
print("  petal_len × petal_wid = +0.96 ◄ correlação muito alta!")
print("    → flores com pétalas compridas tendem a ter pétalas largas também")
print("    → essas duas colunas carregam informação parecida (redundância)")
print()
print("  sepal_wid × petal_len = -0.42")
print("    → correlação negativa moderada")
print("    → flores com sépalas largas tendem a ter pétalas menores")

print()
print("Correlação com o TARGET (espécie):")
corr_target = df[numeric_cols + ["target"]].corr()["target"].drop("target")
for col, val in corr_target.sort_values(key=abs, ascending=False).items():
    forca = ""
    if abs(val) > 0.7:
        forca = "← forte preditora ◄"
    elif abs(val) > 0.4:
        forca = "← moderada"
    else:
        forca = "← fraca"
    print(f"  {col:<12} correlação com target: {val:>+.4f}  {forca}")

print()
print("=" * 60)
print("RESUMO DO EDA")
print("=" * 60)
print()
print("  [1] VISÃO GERAL:  150 amostras, 4 features numéricas, 3 classes")
print("  [2] MISSING:      0 valores nulos → dataset limpo")
print("  [3] TARGET:       Balanceado → 50 por classe (33.3% cada)")
print("  [4] OUTLIERS:     Poucos outliers (dataset de qualidade)")
print("  [+] CORRELAÇÕES:  petal_len e petal_wid altamente correlacionadas")
print("                    petal_len é a feature mais correlacionada com target")
print()
print("Próximo passo: com esse entendimento dos dados, você pode treinar")
print("o modelo com mais confiança. Rode o train.py para ver o resultado!")
print()
print("=" * 60)
