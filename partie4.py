import pandas as pd
import numpy as np
import random

# CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES


# Chargement du fichier CSV
df = pd.read_csv('airnova_flights.csv')

# Conversion des dates
date_columns = ['scheduled_departure', 'scheduled_arrival']
for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Feature Engineering (variables créées dans les parties précédentes)
df['is_delayed'] = (df['delay_minutes'] > 15).astype(int)
df['departure_hour'] = df['scheduled_departure'].dt.hour
df['is_peak_hours'] = ((df['departure_hour'] >= 7) & (df['departure_hour'] <= 9) | 
                        (df['departure_hour'] >= 17) & (df['departure_hour'] <= 19)).astype(int)
df['passenger_density'] = df['passengers'] / df['distance_km']

print("✓ Données chargées et prétraitées")
print(f"Shape: {df.shape}")

# PRÉPARATION DES DONNÉES
#
# Label Encoding pour aircraft_type (ordinal car certains appareils sont "meilleurs")
print("\n1. aircraft_type → Label Encoding")
print("   Justification: Il y a un ordre logique basé sur la taille/complexité des appareils")
print("   Les appareils régionaux (Embraer) < moyen-courriers (A320, 737) < long-courriers")
print(f"   Catégories: {list(df['aircraft_type'].unique())}")

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
print(f"   Colonnes créées: {list(day_dummies.columns)}")

# Label Encoding pour cabin_class
print("\n4. cabin_class → Label Encoding")
print("   Justification: Il y a un ordre naturel: Économique < Affaires < Première")
print("   Label Encoding est approprié car il préserve cet ordre")
print(f"   Catégories: {list(df['cabin_class'].unique())}")

cabin_mapping = {cabin: idx for idx, cabin in enumerate(df['cabin_class'].unique())}
df['cabin_class_encoded'] = df['cabin_class'].map(cabin_mapping)
print(f"   Encodage: {cabin_mapping}")

# PRÉPARATION DES FEATURES POUR LE MODÈLE
# Liste des features à utiliser
features_to_use = [
    'departure_hour',        # Numérique: Heure de départ
    'is_peak_hours',        # Binaire: Heures de pointe
    'flight_duration_min',  # Numérique: Durée du vol
    'load_factor_pct',      # Numérique: Taux de remplissage
    'passengers',           # Numérique: Nombre de passengers
    'distance_km',          # Numérique: Distance
    'is_weekend',           # Binaire: Week-end
    'is_international',     # Binaire: Vol international
    'aircraft_type_encoded',# Numérique encodé: Type d'appareil
    'cabin_class_encoded'   # Numérique encodé: Classe de cabine
]

# Ajouter les colonnes one-hot pour season
season_cols = [col for col in df.columns if col.startswith('season_')]
features_to_use.extend(season_cols)

# Ajouter les colonnes one-hot pour day_of_week
day_cols = [col for col in df.columns if col.startswith('day_')]
features_to_use.extend(day_cols)

print(f"\nFeatures finales sélectionnées ({len(features_to_use)} variables):")
print("-"*40)
for i, feat in enumerate(features_to_use, 1):
    print(f"  {i:2d}. {feat}")

# Préparation X (features) et y (target)
X = df[features_to_use]
y = df['is_delayed']

print(f"\nShape de X: {X.shape}")
print(f"Shape de y: {y.shape}")

# DIVISION TRAIN/TEST (80/20)

# Implémentation manuelle du train-test split
random.seed(42)  # Pour la reproductibilité

# Créer un index pour le split
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

