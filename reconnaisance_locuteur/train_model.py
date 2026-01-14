import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import OUTPUT_ROOT

print("\n" + "="*70)
print("🎓 SECTION 6 : ENTRAÎNEMENT DU MODÈLE")
print("="*70)

# ========================================
# 1. CHARGEMENT DATASET
# ========================================

print("\n📂 Chargement du dataset...")
dataset_path = OUTPUT_ROOT / 'features' / 'police_radio_dataset.csv'

if not dataset_path.exists():
    print(f"❌ Dataset introuvable: {dataset_path}")
    exit(1)

dataset_df = pd.read_csv(dataset_path)
print(f"✅ Chargé: {len(dataset_df)} échantillons")

# ========================================
# 2. PRÉPARATION
# ========================================

print("\n📊 Préparation des données...")
X = dataset_df.drop('speaker_id', axis=1).values
y = dataset_df['speaker_id'].values

print(f"   Dataset shape: {X.shape}")
print(f"   Agents uniques: {len(np.unique(y))}")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\n🏷️  Encodage des labels:")
print(f"   Exemple: {y[0]} → {y_encoded[0]}")
print(f"   Classes: {len(label_encoder.classes_)}")

# ========================================
# 3. SPLIT (70/15/15)
# ========================================

print(f"\n✂️  Split du dataset...")

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.15, 
    random_state=42, 
    stratify=y_encoded
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, 
    test_size=0.176,
    random_state=42, 
    stratify=y_temp
)

print(f"   Train: {len(X_train):5d} ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Val:   {len(X_val):5d} ({len(X_val)/len(X)*100:.1f}%)")
print(f"   Test:  {len(X_test):5d} ({len(X_test)/len(X)*100:.1f}%)")

num_agents = len(label_encoder.classes_)
print(f"\n⚖️  Balance:")
print(f"   Train: {len(X_train) / num_agents:.1f} samples/agent")
print(f"   Val:   {len(X_val) / num_agents:.1f} samples/agent")
print(f"   Test:  {len(X_test) / num_agents:.1f} samples/agent")

# ========================================
# 4. NORMALISATION
# ========================================

print(f"\n📏 Normalisation StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"   Mean avant: {X_train.mean():.3f}")
print(f"   Mean après: {X_train_scaled.mean():.3f}")
print(f"   Std après:  {X_train_scaled.std():.3f}")

# ========================================
# 5. ENTRAÎNEMENT SVM
# ========================================

print(f"\n" + "="*70)
print("⏳ ENTRAÎNEMENT SVM (kernel RBF)...")
print("="*70)
print(f"⏱️  Estimation: 3-5 minutes pour {num_agents} agents")
print(f"📊 Dataset: {len(X_train):,} × {X.shape[1]} features")
print(f"🎯 Classes: {num_agents}\n")

start_time = time.time()

model = SVC(
    kernel='rbf',
    C=33,
    gamma=0.0045,
    random_state=42,
    verbose=False
)

model.fit(X_train_scaled, y_train)

training_time = time.time() - start_time
print(f"\n✅ Entraînement: {training_time:.1f}s ({training_time/60:.1f} min)")

# ========================================
# 6. PRÉDICTIONS
# ========================================

print(f"\n🔮 Prédictions...")
y_train_pred = model.predict(X_train_scaled)
y_val_pred = model.predict(X_val_scaled)
y_test_pred = model.predict(X_test_scaled)

# ========================================
# 7. MÉTRIQUES
# ========================================

print(f"\n" + "="*70)
print("📊 RÉSULTATS DE PERFORMANCE")
print("="*70)

train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\n🎯 ACCURACY:")
print(f"   Train: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"   Val:   {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"   Test:  {test_acc:.4f} ({test_acc*100:.2f}%)")

print(f"\n💡 INTERPRÉTATION:")
if test_acc >= 0.90:
    print("   ⭐ EXCELLENT!")
elif test_acc >= 0.85:
    print("   ✅ TRÈS BON!")
elif test_acc >= 0.80:
    print("   ✅ BON!")
else:
    print("   ⚠️  MOYEN")

overfitting = train_acc - test_acc
print(f"\n🔍 Écart Train-Test: {overfitting:.4f} ({overfitting*100:.2f}%)")
if overfitting < 0.05:
    print("   ✅ Pas d'overfitting")
elif overfitting < 0.10:
    print("   ⚠️  Léger overfitting")
else:
    print("   ❌ Overfitting important")

# ========================================
# 8. MATRICE DE CONFUSION
# ========================================

print(f"\n📊 Matrice de confusion...")

cm = confusion_matrix(y_test, y_test_pred)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

sns.heatmap(cm, cmap='Blues', cbar=True, square=True, 
            xticklabels=False, yticklabels=False, ax=axes[0],
            annot=False, fmt='d')
axes[0].set_title(f'Matrice - {num_agents} Agents\nTest: {test_acc:.2%}', 
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel('Prédit', fontsize=12)
axes[0].set_ylabel('Réel', fontsize=12)

diagonal_accuracy = np.diag(cm).sum() / cm.sum()
axes[0].text(0.5, -0.1, f'Diagonale: {diagonal_accuracy:.2%}', 
             transform=axes[0].transAxes, ha='center', fontsize=11)

error_rates = 100 - np.diag(cm_percent)
axes[1].hist(error_rates, bins=20, color='coral', edgecolor='darkred', alpha=0.7)
axes[1].axvline(error_rates.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Moy: {error_rates.mean():.1f}%')
axes[1].set_title('Distribution Erreurs', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Taux erreur (%)', fontsize=12)
axes[1].set_ylabel('Agents', fontsize=12)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()

perf_path = OUTPUT_ROOT / 'models' / 'model_performance_50agents.png'
plt.savefig(perf_path, dpi=150, bbox_inches='tight')
print(f"   ✅ {perf_path}")
plt.show()
plt.close()

# ========================================
# 9. RAPPORT DÉTAILLÉ
# ========================================

print(f"\n📋 RAPPORT:")
report = classification_report(y_test, y_test_pred, 
                               target_names=label_encoder.classes_,
                               output_dict=True, zero_division=0)

report_df = pd.DataFrame(report).T
report_df = report_df[report_df.index.str.startswith('id')]
report_df = report_df.sort_values('f1-score', ascending=False)

print("\n🏆 TOP 5:")
print(report_df.head(5)[['precision', 'recall', 'f1-score', 'support']].to_string())

print("\n⚠️  WORST 5:")
print(report_df.tail(5)[['precision', 'recall', 'f1-score', 'support']].to_string())

# ========================================
# 10. SAUVEGARDE
# ========================================

print(f"\n💾 Sauvegarde...")

models_dir = OUTPUT_ROOT / 'models'
models_dir.mkdir(exist_ok=True)

model_path = models_dir / 'svm_police_radio_model.pkl'
scaler_path = models_dir / 'scaler.pkl'
encoder_path = models_dir / 'label_encoder.pkl'

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(label_encoder, encoder_path)

print(f"   ✅ {model_path.name}")
print(f"   ✅ {scaler_path.name}")
print(f"   ✅ {encoder_path.name}")

print("\n" + "="*70)
print("🎉 SECTION 6 TERMINÉE!")
print("="*70)

print(f"\n📊 RÉSUMÉ:")
print(f"   Agents: {len(label_encoder.classes_)}")
print(f"   Features: {X.shape[1]}")
print(f"   Test Accuracy: {test_acc:.2%}")
print(f"   Temps: {training_time:.1f}s")
print(f"   Fichiers:")
print(f"    - svm_police_radio_model.pkl")
print(f"    - scaler.pkl")
print(f"    - label_encoder.pkl")
print(f"    - model_performance_50agents.png")
print("="*70 + "\n")
