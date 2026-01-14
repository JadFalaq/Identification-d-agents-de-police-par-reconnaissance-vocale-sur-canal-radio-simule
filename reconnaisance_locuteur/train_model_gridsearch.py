import numpy as np
import pandas as pd
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from config import OUTPUT_ROOT

print("="*70)
print("🔍 OPTIMISATION HYPERPARAMÈTRES - GRID SEARCH")
print("="*70)


print("\n📂 Chargement du dataset...")
dataset_path = OUTPUT_ROOT / 'features' / 'police_radio_dataset.csv'

if not dataset_path.exists():
    print(f"❌ Dataset introuvable: {dataset_path}")
    print("   → Exécutez d'abord: python build_dataset.py")
    exit(1)

dataset_df = pd.read_csv(dataset_path)
print(f"✅ Dataset chargé: {len(dataset_df)} échantillons")

# ========================================
# 2. PRÉPARATION DONNÉES
# ========================================

print("\n📊 Préparation des données...")

X = dataset_df.drop('speaker_id', axis=1).values
y = dataset_df['speaker_id'].values

print(f"   Dataset shape: {X.shape}")
print(f"   Agents uniques: {len(np.unique(y))}")

# Encoder labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\n🏷️  Encodage des labels:")
print(f"   Classes: {len(label_encoder.classes_)}")

# ========================================
# 3. SPLIT STRATIFIÉ (70/15/15)
# ========================================

print(f"\n✂️  Split du dataset...")

# Premier split : 85% / 15% (test)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.15, 
    random_state=42, 
    stratify=y_encoded
)

# Deuxième split : train / val
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, 
    test_size=0.176,  # 0.176 * 0.85 ≈ 0.15
    random_state=42, 
    stratify=y_temp
)

print(f"   Train: {len(X_train):5d} ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Val:   {len(X_val):5d} ({len(X_val)/len(X)*100:.1f}%)")
print(f"   Test:  {len(X_test):5d} ({len(X_test)/len(X)*100:.1f}%)")

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
# 5. GRID SEARCH - OPTIMISATION
# ========================================

print("\n" + "="*70)
print("🔎 GRID SEARCH - RECHERCHE HYPERPARAMÈTRES OPTIMAUX")
print("="*70)

# Stratified K-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grille de paramètres à tester
param_grid = {
    "C": [33, 32],
    "gamma": [0.0045],
    "kernel": ["rbf"]
}

print("\n📋 Grille de recherche:")
print(f"   C: {param_grid['C']}")
print(f"   Gamma: {param_grid['gamma']}")
print(f"   Kernel: {param_grid['kernel']}")
print(f"\n   Total combinaisons: {len(param_grid['C']) * len(param_grid['gamma'])}")
print(f"   CV Folds: 5")
print(f"   Total entraînements: {len(param_grid['C']) * len(param_grid['gamma']) * 5}")

print("\n⏳ Recherche en cours (durée estimée: 10-20 min)...\n")

base_svm = SVC()

grid = GridSearchCV(
    estimator=base_svm,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,  # Utilise tous les CPU disponibles
    verbose=2    # Affiche progression détaillée
)

start_time = time.time()
grid.fit(X_train_scaled, y_train)
search_time = time.time() - start_time

print(f"\n✅ Recherche terminée en {search_time/60:.1f} minutes")

# ========================================
# 6. RÉSULTATS GRID SEARCH
# ========================================

print("\n" + "="*70)
print("📊 RÉSULTATS GRID SEARCH")
print("="*70)

print(f"\n🏆 Meilleurs paramètres trouvés:")
print(f"   C: {grid.best_params_['C']}")
print(f"   Gamma: {grid.best_params_['gamma']}")
print(f"   Kernel: {grid.best_params_['kernel']}")

print(f"\n🎯 Best CV accuracy: {grid.best_score_:.4f} ({grid.best_score_*100:.2f}%)")

# Afficher top 5 combinaisons
print("\n📈 Top 5 combinaisons:")
results_df = pd.DataFrame(grid.cv_results_)
results_df = results_df.sort_values('rank_test_score')

for idx, row in results_df.head(5).iterrows():
    print(f"   {int(row['rank_test_score'])}. C={row['param_C']}, "
          f"gamma={row['param_gamma']}, "
          f"accuracy={row['mean_test_score']:.4f} "
          f"(±{row['std_test_score']:.4f})")

best_model = grid.best_estimator_

# ========================================
# 7. ÉVALUATION SUR VALIDATION
# ========================================

print("\n" + "="*70)
print("🎯 ÉVALUATION SUR VALIDATION SET")
print("="*70)

val_pred = best_model.predict(X_val_scaled)
val_acc = accuracy_score(y_val, val_pred)

print(f"\nAccuracy validation: {val_acc:.4f} ({val_acc*100:.2f}%)")

# ========================================
# 8. RÉ-ENTRAÎNEMENT FINAL (train+val)
# ========================================

print("\n" + "="*70)
print("⏳ RÉ-ENTRAÎNEMENT MODÈLE FINAL")
print("="*70)

X_train_full = np.vstack([X_train_scaled, X_val_scaled])
y_train_full = np.concatenate([y_train, y_val])

print(f"\n📊 Dataset final:")
print(f"   Échantillons: {len(X_train_full)}")
print(f"   Features: {X_train_full.shape[1]}")
print(f"   Classes: {len(np.unique(y_train_full))}")

final_model = SVC(
    kernel="rbf",
    C=grid.best_params_["C"],
    gamma=grid.best_params_["gamma"],
    random_state=42,
    verbose=False
)

print("\n⏳ Entraînement...")
start_time = time.time()
final_model.fit(X_train_full, y_train_full)
train_time = time.time() - start_time

print(f"✅ Entraînement terminé en {train_time:.1f}s")

# ========================================
# 9. ÉVALUATION FINALE SUR TEST
# ========================================

print("\n" + "="*70)
print("🏁 ÉVALUATION FINALE SUR TEST SET")
print("="*70)

test_pred = final_model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, test_pred)

print(f"\n🎯 Accuracy TEST FINALE: {test_acc:.4f} ({test_acc*100:.2f}%)")

# ========================================
# 10. COMPARAISON PERFORMANCES
# ========================================

print("\n" + "="*70)
print("📊 COMPARAISON DES PERFORMANCES")
print("="*70)

print(f"\n{'Ensemble':<20} {'Accuracy':<15} {'Échantillons':<15}")
print("-" * 50)
print(f"{'CV (train)':<20} {grid.best_score_:.4f} ({grid.best_score_*100:.2f}%)   {len(X_train)}")
print(f"{'Validation':<20} {val_acc:.4f} ({val_acc*100:.2f}%)   {len(X_val)}")
print(f"{'Test (final)':<20} {test_acc:.4f} ({test_acc*100:.2f}%)   {len(X_test)}")

# Diagnostic
gap_train_test = grid.best_score_ - test_acc
print(f"\n🔍 Écart CV-Test: {gap_train_test:.4f} ({gap_train_test*100:.2f}%)")

if gap_train_test < 0.05:
    print("   ✅ Excellent! Pas d'overfitting")
elif gap_train_test < 0.10:
    print("   ⚠️  Léger overfitting (acceptable)")
else:
    print("   ❌ Overfitting détecté")

# ========================================
# 11. SAUVEGARDE DU MODÈLE OPTIMISÉ
# ========================================

print("\n" + "="*70)
print("💾 SAUVEGARDE DU MODÈLE OPTIMISÉ")
print("="*70)

models_dir = OUTPUT_ROOT / 'models'
models_dir.mkdir(exist_ok=True)

model_path = models_dir / 'svm_police_radio_model_optimized.pkl'
scaler_path = models_dir / 'scaler_optimized.pkl'
encoder_path = models_dir / 'label_encoder_optimized.pkl'

joblib.dump(final_model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(label_encoder, encoder_path)

print(f"\n✅ Fichiers sauvegardés:")
print(f"   📁 {model_path.name}")
print(f"   📁 {scaler_path.name}")
print(f"   📁 {encoder_path.name}")

# Sauvegarder les résultats du Grid Search
results_path = models_dir / 'grid_search_results.csv'
results_df.to_csv(results_path, index=False)
print(f"   📁 {results_path.name}")

# ========================================
# 12. RÉSUMÉ FINAL
# ========================================

print("\n" + "="*70)
print("🎉 OPTIMISATION TERMINÉE!")
print("="*70)

print(f"\n📊 RÉSUMÉ FINAL:")
print(f"   Meilleurs hyperparamètres:")
print(f"      C: {grid.best_params_['C']}")
print(f"      Gamma: {grid.best_params_['gamma']}")
print(f"   ")
print(f"   Performances:")
print(f"      CV Accuracy: {grid.best_score_*100:.2f}%")
print(f"      Validation Accuracy: {val_acc*100:.2f}%")
print(f"      Test Accuracy: {test_acc*100:.2f}%")
print(f"   ")
print(f"   Temps total: {search_time/60 + train_time/60:.1f} min")
print(f"   Agents: {len(label_encoder.classes_)}")
print(f"   Features: {X.shape[1]}")

print("\n" + "="*70)
