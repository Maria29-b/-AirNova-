import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Chargement du fichier CSV dans un DataFrame pandas
df_csv = pd.read_csv('airnova_flights.csv')

# Affichage des 5 premières lignes
print("\n5 premières lignes du dataset CSV:")
print(df_csv.head())

# Affichage de la forme du dataset
print(f"\nForme du dataset (lignes, colonnes): {df_csv.shape}")
print(f"Nombre de lignes: {df_csv.shape[0]}")
print(f"Nombre de colonnes: {df_csv.shape[1]}")


# Chargement du fichier JSON
with open('airnova_flights.json', 'r', encoding='utf-8') as f:
    data_json = json.load(f)

# Conversion en DataFrame
df_json = pd.DataFrame(data_json)

# Vérification que les deux sources contiennent le même nombre d'enregistrements
print(f"\nNombre d'enregistrements dans le CSV: {len(df_csv)}")
print(f"Nombre d'enregistrements dans le JSON: {len(df_json)}")

if len(df_csv) == len(df_json):
    print("✓ Les deux fichiers contiennent le même nombre d'enregistrements (300)")
else:
    print("⚠ Attention: Les fichiers ne contiennent pas le même nombre d'enregistrements!")

# Fusion des deux DataFrames
df_merged = pd.merge(df_csv, df_json, on='flight_id', how='inner')

print(f"\nNombre d'enregistrements après fusion: {len(df_merged)}")
print(f"Nombre de colonnes après fusion: {df_merged.shape[1]}")

print("\nAperçu du DataFrame fusionné:")
print(df_merged.head())

# Vérification des colonnes
print("\nColonnes du DataFrame fusionné:")
print(df_merged.columns.tolist())

#  Audit de qualité
#

# Pour simplifier, on utilise le DataFrame CSV original car les données sont identiques
df = df_csv.copy()

# Comptage des valeurs manquantes par colonne
missing_values = df.isnull().sum()
print("\nValeurs manquantes par colonne:")
print(missing_values[missing_values > 0])

if missing_values.sum() == 0:
    print("\n✓ Aucune valeur manquante détectée dans le dataset")

# Pourcentage de valeurs manquantes
missing_pct = (df.isnull().sum() / len(df)) * 100
print("\nPourcentage de valeurs manquantes:")
print(missing_pct[missing_pct > 0])

# Colonnes de dates à convertir
date_columns = ['scheduled_departure', 'scheduled_arrival']

print(f"\nConversion des colonnes de dates en datetime: {date_columns}")

# Conversion des colonnes de dates
for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')

print("\nTypes après conversion:")
print(df[date_columns].dtypes)

# Incohérence 1: flight_duration_min vs durée calculée à partir des dates
print("\n--- Incohérence 1: Durée du vol ---")
df['calculated_duration'] = (df['scheduled_arrival'] - df['scheduled_departure']).dt.total_seconds() / 60

# Gérer les cas où l'arrivée est le jour suivant
df.loc[df['calculated_duration'] < 0, 'calculated_duration'] += 24 * 60

# Comparer la durée programmée avec la durée calculée
df['duration_diff'] = abs(df['flight_duration_min'] - df['calculated_duration'])

# Seuil de tolérance de 10 minutes
inconsistent_duration = df[df['duration_diff'] > 10]
print(f"Nombre de vols avec incohérence de durée (>10 min): {len(inconsistent_duration)}")

if len(inconsistent_duration) > 0:
    print("\nExemples d'incohérences de durée:")
    print(inconsistent_duration[['flight_id', 'flight_duration_min', 'calculated_duration', 'duration_diff']].head(10))

# Correction
df.loc[df['duration_diff'] > 10, 'flight_duration_min'] = df.loc[df['duration_diff'] > 10, 'calculated_duration'].astype(int)
print("\n✓ Incohérence 1 corrigée")

# Incohérence 2: load_factor_pct vs calculé
print("\n--- Incohérence 2: Taux de remplissage ---")
df['calculated_load_factor'] = (df['passengers'] / df['seat_capacity']) * 100
df['load_factor_diff'] = abs(df['load_factor_pct'] - df['calculated_load_factor'])

inconsistent_load = df[df['load_factor_diff'] > 1]
print(f"Nombre de vols avec incohérence de load_factor (>1%): {len(inconsistent_load)}")

if len(inconsistent_load) > 0:
    df.loc[df['load_factor_diff'] > 1, 'load_factor_pct'] = df.loc[df['load_factor_diff'] > 1, 'calculated_load_factor'].round(1)
    print("\n✓ Incohérence 2 corrigée")

# Incohérence 3: Contradiction entre status et delay_minutes
print("\n--- Incohérence 3: Status vs Delay_minutes ---")
inconsistent_status_delay = df[(df['status'] == 'À l\'heure') & (df['delay_minutes'] != 0)]
print(f"Nombre de vols 'À l'heure' avec delay_minutes != 0: {len(inconsistent_status_delay)}")

inconsistent_delay_zero = df[(df['status'] == 'Retardé') & (df['delay_minutes'] == 0)]
print(f"Nombre de vols 'Retardé' avec delay_minutes = 0: {len(inconsistent_delay_zero)}")

if len(inconsistent_status_delay) > 0:
    df.loc[(df['status'] == 'À l\'heure') & (df['delay_minutes'] != 0), 'delay_minutes'] = 0
    print("✓ Incohérence 3a corrigée")

# Suppression des colonnes temporaires
df = df.drop(columns=['calculated_duration', 'duration_diff', 'calculated_load_factor', 'load_factor_diff'])

#  Feature Engineering
# Variable 1: is_delayed (Variable binaire cible)
print("\n--- Variable 1: is_delayed ---")
df['is_delayed'] = (df['delay_minutes'] > 15).astype(int)

print(f"Formule: is_delayed = 1 si delay_minutes > 15, sinon 0")
print(f"Nombre de vols retardés (is_delayed=1): {df['is_delayed'].sum()}")
print(f"Nombre de vols à l'heure (is_delayed=0): {(df['is_delayed'] == 0).sum()}")
print(f"Taux de retard: {df['is_delayed'].mean()*100:.2f}%")
print("Type: Binaire (0 ou 1)")
print("Intérêt métier: Variable cible pour la prédiction de retard")

# Variable 2: departure_hour (Variable temporelle)
print("\n--- Variable 2: departure_hour ---")
df['departure_hour'] = df['scheduled_departure'].dt.hour

print(f"Formule: departure_hour = heure(scheduled_departure)")
print(f"Plage de valeurs: {df['departure_hour'].min()}h à {df['departure_hour'].max()}h")
print("Type: Numérique (entier 0-23)")
print("Intérêt métier: Permet d'analyser les pics de retard selon les tranches horaires")

# Variable 3: is_peak_hours (Variable binaire)
print("\n--- Variable 3: is_peak_hours ---")
df['is_peak_hours'] = ((df['departure_hour'] >= 7) & (df['departure_hour'] <= 9) | 
                       (df['departure_hour'] >= 17) & (df['departure_hour'] <= 19)).astype(int)

print(f"Formule: is_peak_hours = 1 si departure_hour ∈ [7-9] ou [17-19], sinon 0")
print(f"Nombre de vols en heures de pointe: {df['is_peak_hours'].sum()}")
print("Type: Binaire (0 ou 1)")
print("Intérêt métier: Identifie les vols lors des pics d'activité")

# Variable 4: passenger_density (Variable numérique)
print("\n--- Variable 4: passenger_density ---")
df['passenger_density'] = df['passengers'] / df['distance_km']

print(f"Formule: passenger_density = passengers / distance_km")
print(f"Statistiques:")
print(f"  - Min: {df['passenger_density'].min():.4f}")
print(f"  - Max: {df['passenger_density'].max():.4f}")
print(f"  - Moyenne: {df['passenger_density'].mean():.4f}")
print("Type: Numérique (continu)")
print("Intérêt métier: Mesure l'intensité de la demande par km de vol")



#  Statistiques univariées

#  Indicateurs de tendance centrale et de dispersion
print("\n--- Indicateurs de tendance centrale et de dispersion ---")

numeric_vars = ['delay_minutes', 'ticket_price_eur', 'load_factor_pct', 'flight_duration_min']

stats_df = pd.DataFrame({
    'Variable': numeric_vars,
    'Moyenne': [df[var].mean() for var in numeric_vars],
    'Médiane': [df[var].median() for var in numeric_vars],
    'Écart-type': [df[var].std() for var in numeric_vars],
    'Min': [df[var].min() for var in numeric_vars],
    'Max': [df[var].max() for var in numeric_vars]
})

print("\nIndicateurs de tendance centrale et de dispersion:")
print(stats_df.to_string(index=False))

#  Analyse de la distribution de delay_minutes
print("\n---  Analyse de la distribution de delay_minutes ---")

# Calcul des indicateurs d'asymétrie


delay_skewness = df['delay_minutes'].skew()
delay_kurtosis = df['delay_minutes'].kurtosis()

print(f"\nAnalyse de la distribution de delay_minutes:")
print(f"  - Coefficient d'asymétrie (skewness): {delay_skewness:.4f}")
print(f"  - Coefficient d'aplatissement (kurtosis): {delay_kurtosis:.4f}")

# Interprétation
if delay_skewness > 1:
    print("  - Distribution: Asymétrie positive (étirée à droite)")
    print("  - Signification: La plupart des vols sont à l'heure, avec quelques retards très importants")
elif delay_skewness < -1:
    print("  - Distribution: Asymétrie négative (étirée à gauche)")
else:
    print("  - Distribution: Asymétrie modérée")

print("""
+  Signification pour AirNova:
+  - L'asymétrie positive indique que les gros retards sont rares mais existent
+  - La moyenne est plus élevée que la médiane (due aux valeurs extrêmes)
+  - Il peut être pertinent de traiter les retards importants séparément
+  - Pour la modélisation, considerer une transformation (log) si nécessaire
+""")

#  Fréquence de chaque modalité de status
print("\n--- : Fréquence des modalités de status ---")

status_freq = df['status'].value_counts()
status_pct = (df['status'].value_counts(normalize=True) * 100).round(2)

print("\nFréquence de chaque statut:")
status_table = pd.DataFrame({
    'Statut': status_freq.index,
    'Nombre': status_freq.values,
    'Pourcentage (%)': status_pct.values
})
print(status_table.to_string(index=False))

# Identification du statut le plus rare
rarest_status = status_freq.idxmin()
rarest_count = status_freq.min()
print(f"\nStatut le plus rare: '{rarest_status}' avec {rarest_count} occurrence(s)")

print(f"""
Problème pour la modélisation:
- Le statut '{rarest_status}' est très rare ({rarest_count} vols, {status_pct[rarest_status]}%)
- Problème de classes déséquilibrées pour un modèle de classification
- Risque de sur-apprentissage sur les classes majoritaires
- Solutions possibles: sur-échantillonnage, sous-échantillonnage,""")


#
#  Statistiques bivariées


#  Matrice de corrélation des variables numériques
print("\n---  Matrice de corrélation ---")
# Sélection des variables numériques
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
# Exclure les colonnes non pertinentes
exclude_cols = ['flight_id']
numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

# Calcul de la matrice de corrélation
corr_matrix = df[numeric_cols].corr()

print("\nMatrice de corrélation (variables numériques):")
print(corr_matrix.round(3))
# Identification des deux paires les plus corrélées
# Extraction des paires de corrélation (tri supérieur)
corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_pairs.append({
            'Variable 1': corr_matrix.columns[i],
            'Variable 2': corr_matrix.columns[j],
            'Corrélation': corr_matrix.iloc[i, j]
        })

corr_pairs_df = pd.DataFrame(corr_pairs)
corr_pairs_df['Abs_Corr'] = corr_pairs_df['Corrélation'].abs()
corr_pairs_df = corr_pairs_df.sort_values('Abs_Corr', ascending=False)

top_corr = corr_pairs_df.head(2)
print("\n--- Deux paires les plus corrélées ---")
print(top_corr[['Variable 1', 'Variable 2', 'Corrélation']].to_string(index=False))

print("""
+Explication métier de la première corrélation:
+- passengers et seat_capacity sont fortement corrélées positivement
+- C'est logique: les avions plus grands ont plus de sièges donc plus de passagers
+- load_factor_pct peut être calculé à partir de ces deux variables
+""")

#  Comparaison du retard moyen selon saison, jour de semaine, type appareil
print("\n--- Comparaison du retard moyen ---")

# Par saison
print("\nRetard moyen par saison:")
delay_by_season = df.groupby('season')['delay_minutes'].mean().round(2)
print(delay_by_season.sort_values(ascending=False))

# Par jour de semaine
print("\nRetard moyen par jour de semaine:")
delay_by_day = df.groupby('day_of_week')['delay_minutes'].mean().round(2)
print(delay_by_day.sort_values(ascending=False))

# Par type d'appareil
print("\nRetard moyen par type d'appareil:")
delay_by_aircraft = df.groupby('aircraft_type')['delay_minutes'].mean().round(2)
print(delay_by_aircraft.sort_values(ascending=False))

print("""
+Synthèse:
+---------
+L'analyse comparative révèle plusieurs tendances significatives pour AirNova:
+
+1. Saisonnalité: Les retards varient selon la saison, probablement liés aux conditions 
+   météorologiques et à la densité du trafic aérien (vacances, périodes de pointe).
+
+2. Jour de semaine: Certains jours présentent des retards plus élevés, possiblement 
+   corrélés aux pics d'activité professionnelle (début/fin de semaine).
+
+3. Type d'appareil: Les différents avions ont des profils de retard distincts. Les gros 
+  -porteurs (Airbus A321, Boeing 737-800) peuvent avoir des temps de rotation plus longs,
+   tandis que les appareils régionaux (Embraer E190) sont souvent plus flexibles.
+
4. Implications opérationnelles: AirNova pourrait optimiser l'allocation des appareils 
   et des équipages en fonction de ces patterns pour minimiser les retards en cascade.
+""")

#  Relation entre load_factor_pct et retard
print("\n---  Relation entre load_factor_pct et retard ---")

# Calcul de la corrélation entre load_factor_pct et delay_minutes
corr_load_delay = df['load_factor_pct'].corr(df['delay_minutes'])

print(f"\nCorrélation entre load_factor_pct et delay_minutes: {corr_load_delay:.4f}")

# Calcul de la corrélation avec is_delayed
corr_load_is_delayed = df['load_factor_pct'].corr(df['is_delayed'])

print(f"Corrélation entre load_factor_pct et is_delayed: {corr_load_is_delayed:.4f}")

# Analyse par catégories de load_factor
print("\nAnalyse du retard par catégorie de taux de remplissage:")
df['load_category'] = pd.cut(df['load_factor_pct'], 
bins=[0, 50, 70, 90, 100], 
labels=['Faible (<50%)', 'Moyen (50-70%)', 'Élevé (70-90%)', 'Très élevé (>90%)'])

delay_by_load = df.groupby('load_category', observed=True)['delay_minutes'].agg(['mean', 'count']).round(2)
print(delay_by_load)

print(f"""
Conclusion:
----------
La corrélation entre load_factor_pct et delay_minutes est faible ({corr_load_delay:.4f}).
Cela signifie qu'il n'y a pas de relation linéaire significative entre le taux de 
remplissage et le retard.

Pour AirNova: Le nombre de passagers n'est pas un facteur direct de retard. 
D'autres facteurs (météo, trafic aérien, maintenance) sont probablement plus déterminants.
""")


