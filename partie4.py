import pandas as pd
import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES


# Chargement du fichier CSV
df = pd.read_csv('airnova_flights.csv')

# Conversion des dates
date_columns = ['scheduled_departure', 'scheduled_arrival']
for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Feature Engineering
df['is_delayed'] = (df['delay_minutes'] > 15).astype(int)
df['departure_hour'] = df['scheduled_departure'].dt.hour
df['is_peak_hours'] = ((df['departure_hour'] >= 7) & (df['departure_hour'] <= 9) | 
                        (df['departure_hour'] >= 17) & (df['departure_hour'] <= 19)).astype(int)
df['passenger_density'] = df['passengers'] / df['distance_km']

print("✓ Données chargées et prétraitées")
print(f"Shape: {df.shape}")

# PRÉPARATION DES DONNÉES - Encodage
# Label Encoding pour aircraft_type
aircraft_mapping = {aircraft: idx for idx, aircraft in enumerate(df['aircraft_type'].unique())}
df['aircraft_type_encoded'] = df['aircraft_type'].map(aircraft_mapping)
print(f"   Encodage: {aircraft_mapping}")

# One-Hot Encoding pour season
print("\n2. season → One-Hot Encoding")
print("   Justification: Pas d'ordre entre les saisons.Toutes les saisons sont égales.")
print("   One-Hot Encoding preferé pour éviter d'impliquer un ordre fictif (Hiver<Printemps<Été<Automne)")
print(f"   Catégories: {list(df['season'].unique())}")

season_dummies = pd.get_dummies(df['season'], prefix='season')
df = pd.concat([df, season_dummies], axis=1)
print(f"   Colonnes créées: {list(season_dummies.columns)}")

# One-Hot Encoding pour day_of_week
print("\n3. day_of_week → One-Hot Encoding")
print("   Justification: Pas d'ordre entre les jours de la semaine")
print("   Monday, Tuesday, Wednesday... n'ont pas d'ordre naturel")
print(f"   Catégories: {list(df['day_of_week'].unique())}")

day_dummies = pd.get_dummies(df['day_of_week'], prefix='day')
df = pd.concat([df, day_dummies], axis=1)


# Label Encoding pour cabin_class
print("\n4. cabin_class → Label Encoding")
print("   Justification: Il y a un ordre naturel: Économique < Affaires < Première")
print("   Label Encoding est approprié car il préserve cet ordre")
print(f"   Catégories: {list(df['cabin_class'].unique())}")

cabin_mapping = {cabin: idx for idx, cabin in enumerate(df['cabin_class'].unique())}
df['cabin_class_encoded'] = df['cabin_class'].map(cabin_mapping)
print(f"   Encodage: {cabin_mapping}")

# Supprimer les colonnes catégorielles originales après encodage
df = df.drop(columns=['day_of_week', 'season', 'aircraft_type', 'cabin_class', 
                       'origin_code', 'origin_city', 'origin_country',
                       'destination_code', 'destination_city', 'destination_country',
                       'flight_number', 'airline', 'status', 
                       'delay_reason', 'cancellation_reason'])


# PRÉPARATION DES FEATURES POUR LE MODÈLE
# Liste des features à utiliser
features_to_use = [
    'departure_hour', 'is_peak_hours', 'flight_duration_min',
    'load_factor_pct', 'passengers', 'distance_km',
    'is_weekend', 'is_international', 'aircraft_type_encoded', 'cabin_class_encoded'
]

# Ajouter les colonnes one-hot pour season
season_cols = [col for col in df.columns if col.startswith('season_')]
features_to_use.extend(season_cols)

# Ajouter les colonnes one-hot pour day_of_week
day_cols = [col for col in df.columns if col.startswith('day_')]
features_to_use.extend(day_cols)

# Préparation X et y
X = df[features_to_use]
y = df['is_delayed']

# Division Train/Test
random.seed(42)
indices = list(range(len(df)))
random.shuffle(indices)

# 80% train, 20% test
split_idx = int(len(indices) * 0.8)
train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

# Appliquer le split
X_train = X.iloc[train_indices].reset_index(drop=True)
X_test = X.iloc[test_indices].reset_index(drop=True)
y_train = y.iloc[train_indices].reset_index(drop=True)
y_test = y.iloc[test_indices].reset_index(drop=True)

print(f"\nShape X_train: {X_train.shape}")
print(f"Shape X_test: {X_test.shape}")

# 
# ENTRAÎNEMENT ET COMPARAISON DE MODÈLES


# Entraînement des 3 modèles et métriques
# Initialisation des modèles
dt_model = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
lr_model = LogisticRegression(max_iter=1000, random_state=42)

# Entraînement
print("\nEntraînement des modèles...")
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)


# Prédictions
y_pred_dt = dt_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_lr = lr_model.predict(X_test)

# Calcul des métriques
def calculate_metrics(y_true, y_pred, model_name):
    return {
        'Modèle': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0)
    }

metrics_dt = calculate_metrics(y_test, y_pred_dt, 'Arbre de Décision')
metrics_rf = calculate_metrics(y_test, y_pred_rf, 'Random Forest')
metrics_lr = calculate_metrics(y_test, y_pred_lr, 'Régression Logistique')

# Tableau synthétique
print("TABLEAU SYNTHÉTIQUE DES MÉTRIQUES")
print(f"{'Modèle':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("-"*60)
print(f"{metrics_dt['Modèle']:<25} {metrics_dt['Accuracy']:>10.4f} {metrics_dt['Precision']:>10.4f} {metrics_dt['Recall']:>10.4f} {metrics_dt['F1-Score']:>10.4f}")
print(f"{metrics_rf['Modèle']:<25} {metrics_rf['Accuracy']:>10.4f} {metrics_rf['Precision']:>10.4f} {metrics_rf['Recall']:>10.4f} {metrics_rf['F1-Score']:>10.4f}")
print(f"{metrics_lr['Modèle']:<25} {metrics_lr['Accuracy']:>10.4f} {metrics_lr['Precision']:>10.4f} {metrics_lr['Recall']:>10.4f} {metrics_lr['F1-Score']:>10.4f}")
print("-"*60)

# Meilleur modèle
models_metrics = [metrics_dt, metrics_rf, metrics_lr]
best_model_info = max(models_metrics, key=lambda x: x['F1-Score'])
best_model_name = best_model_info['Modèle']

print(f"\n✓ Meilleur modèle (basé sur F1-Score): {best_model_name}")
print(f"  - Accuracy: {best_model_info['Accuracy']:.4f}")
print(f"  - Precision: {best_model_info['Precision']:.4f}")
print(f"  - Recall: {best_model_info['Recall']:.4f}")
print(f"  - F1-Score: {best_model_info['F1-Score']:.4f}")


#  Matrice de confusion du meilleur modèle
# Sélectionner les prédictions du meilleur modèle
if best_model_name == 'Arbre de Décision':
    y_pred_best = y_pred_dt
    best_model = dt_model
elif best_model_name == 'Random Forest':
    y_pred_best = y_pred_rf
    best_model = rf_model
else:
    y_pred_best = y_pred_lr
    best_model = lr_model

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred_best)

print("\nMatrice de confusion:")
print(f"                    Prédit")
print(f"                 Non retardé | Retardé")
print(f"Réel Non retardé     {cm[0,0]:>4}    |   {cm[0,1]:>4}")
print(f"Réel Retardé         {cm[1,0]:>4}    |   {cm[1,1]:>4}")

TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
TP = cm[1, 1]

print(f"""
DÉTAIL DES ERREURS:
- Vrais Négatifs (TN): {TN} - Vol correctement prédit comme non retardé
- Faux Positifs (FP): {FP} - Vol prédit retardé mais était à l'heure
- Faux Négatifs (FN): {FN} - Vol prédit à l'heure mais était retardé
- Vrais Positifs (TP): {TP} - Vol correctement prédit comme retardé

TYPE D'ERREURS LE PLUS COÛTEUX POUR AIRNOVA:
=============================================
Le FAUX NÉGATIF ({FN} cas) est le plus coûteux pour AirNova car:
- C'est un vol qui va être retardé mais le modèle prédit "à l'heure"
- AirNova NE PEUT PAS anticiper ce retard
- Conséquences: Retard en cascade, correspondances manquées, clients mécontents
- Impact financier: Rémédiation client, pénalités, réputation endommagée

Le FAUX POSITIF ({FP} cas) est moins coûteux car:
- C'est une "fausse alerte" - le modèle prédit un retard qui n'existe pas
- AirNova peut quand même préparer des mesures préventives
- Conséquence: Légère sur-optimisation des ressources, mais moins risqué
""")

# Affichage graphique
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non retardé', 'Retardé'],
            yticklabels=['Non retardé', 'Retardé'])
plt.xlabel('Prédit', fontsize=12)
plt.ylabel('Réel', fontsize=12)
plt.title(f'Matrice de confusion - {best_model_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('matrice_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nMatrice de confusion enregistrée: matrice_confusion.png")

 
#  Importance des variables

if best_model_name in ['Arbre de Décision', 'Random Forest']:
    feature_importance = pd.DataFrame({
        'Feature': features_to_use,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 des features les plus influentes:")
    print("-"*40)
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['Feature']:<30} {row['Importance']:.4f}")
    
    top_3 = feature_importance.head(3)['Feature'].tolist()
    
    print(f"""
TOP 3 DES FEATURES LES PLUS INFLUENTES:
=========================================
1. {top_3[0]}
2. {top_3[1]}
3. {top_3[2]}

COHÉRENCE AVEC LES PARTIES 2 ET 3:
===================================
""")
    if 'departure_hour' in top_3:
        print("✓ departure_hour est une feature importante")
        print("  → Cohérent avec l'analyse en Partie 3 (Graphique 5)")
        print("  → Les retards varient selon l'heure de la journée")
    
    if 'is_peak_hours' in top_3:
        print("✓ is_peak_hours est une feature importante")
        print("  → Cohérent avec l'analyse en Partie 2")
    
    if 'flight_duration_min' in top_3:
        print("✓ flight_duration_min est une feature importante")
        print("  → Cohérent avec l'analyse en Partie 2 (corrélation)")
    
    if 'passengers' in top_3:
        print("✓ passengers est une feature importante")
        print("  → Cohérent avec l'analyse en Partie 2")
    
    if 'load_factor_pct' in top_3:
        print("✓ load_factor_pct est une feature importante")
        print("  → Cohérent avec l'analyse en Partie 2")
    
    # Graphique
    plt.figure(figsize=(12, 8))
    top_15 = feature_importance.head(15)
    plt.barh(range(len(top_15)), top_15['Importance'].values, color='steelblue')
    plt.yticks(range(len(top_15)), top_15['Feature'].values)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Importance des features - {best_model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nGraphique d'importance des features enregistré: feature_importance.png")
