import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# ==============================================================================
# SAVE MODEL — Serializar o modelo treinado para disco
#
# Suas anotações mencionam os formatos de empacotamento:
#   Pickle / Joblib → Python, scikit-learn, XGBoost
#   ONNX            → interoperabilidade entre frameworks
#   SavedModel      → Keras / TensorFlow
#
# Usamos joblib (preferível ao pickle para sklearn):
#   - Mais eficiente com arrays NumPy (compressão interna)
#   - Mesmo formato, mais performance
#
# IMPORTANTE: Salvar também o scaler junto com o modelo!
# O scaler foi ajustado nos dados de treino. Na inferência, você precisa
# aplicar a MESMA transformação que foi feita durante o treino.
# Se não salvar, os dados novos chegam em escala diferente → predições erradas.
# ==============================================================================

print("=" * 60)
print("SALVANDO MODELO E ARTEFATOS")
print("=" * 60)

# Treino
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)

acc = accuracy_score(y_test, model.predict(X_test_scaled))
f1  = f1_score(y_test, model.predict(X_test_scaled), average="weighted")

print(f"\nModelo treinado:")
print(f"  Accuracy: {acc:.4f}")
print(f"  F1 Score: {f1:.4f}")

# Salvar
joblib.dump(model,  "model.joblib")
joblib.dump(scaler, "scaler.joblib")

print(f"\nArtefatos salvos em disco:")
print(f"  model.joblib  → o modelo treinado")
print(f"  scaler.joblib → o StandardScaler (OBRIGATÓRIO salvar junto)")
print()
print("Por que salvar o scaler separado?")
print("  O scaler.fit() calcula média e std dos dados de TREINO.")
print("  Na inferência usamos scaler.transform() com esses mesmos valores.")
print("  Se recriar o scaler na inferência, os dados chegam em escala errada")
print("  e o modelo dá predições incorretas — bug silencioso e perigoso.")
print()
print("  Analogia: é como salvar a chave junto com a fechadura.")
print("  De nada adianta um sem o outro.")

# Verificar que carregam corretamente
model_carregado  = joblib.load("model.joblib")
scaler_carregado = joblib.load("scaler.joblib")

X_exemplo = np.array([[5.1, 3.5, 1.4, 0.2]])
X_scaled  = scaler_carregado.transform(X_exemplo)
pred      = model_carregado.predict(X_scaled)
proba     = model_carregado.predict_proba(X_scaled)

print(f"\nVerificação: predição para {X_exemplo[0].tolist()}")
print(f"  Classe prevista:   {iris.target_names[pred[0]]} (classe {pred[0]})")
print(f"  Probabilidades:")
for i, nome in enumerate(iris.target_names):
    barra = "█" * int(proba[0][i] * 30)
    print(f"    {nome:<12} {proba[0][i]:.4f}  {barra}")

print()
print("Artefatos prontos para o servidor de inferência (serve.py).")
