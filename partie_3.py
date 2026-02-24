import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# =============================================================================
# CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES
# =============================================================================

# Chargement du fichier CSV
df_csv = pd.read_csv('airnova_flights.csv')
df = df_csv.copy()

# Conversion des dates
date_columns = ['scheduled_departure', 'scheduled_arrival']
for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Feature Engineering (variables nécessaires pour les visualisations)
df['is_delayed'] = (df['delay_minutes'] > 15).astype(int)
df['departure_hour'] = df['scheduled_departure'].dt.hour
df['is_peak_hours'] = ((df['departure_hour'] >= 7) & (df['departure_hour'] <= 9) | 
                        (df['departure_hour'] >= 17) & (df['departure_hour'] <= 19)).astype(int)
df['passenger_density'] = df['passengers'] / df['distance_km']

# Variables numériques pour corrélation
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['flight_id']]

print("✓ Données chargées et prétraitées pour les visualisations")
print(f"Shape: {df.shape}")

# VISUALISATIONS


# Configuration du style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Graphique 1 — Distribution des retards (vols retardés uniquement)
print("\n--- Graphique 1: Distribution des retards ---")

fig1, ax1 = plt.subplots(figsize=(10, 6))

# Filtrer uniquement les vols retardés (delay_minutes > 0)
delayed_flights = df[df['delay_minutes'] > 0]['delay_minutes']

# Histogramme avec kde
sns.histplot(delayed_flights, kde=True, bins=30, color='coral', edgecolor='black', ax=ax1)

# Ajouter la moyenne et la médiane
mean_delay = delayed_flights.mean()
median_delay = delayed_flights.median()

ax1.axvline(mean_delay, color='red', linestyle='--', linewidth=2, label=f'Moyenne: {mean_delay:.1f} min')
ax1.axvline(median_delay, color='blue', linestyle='-', linewidth=2, label=f'Médiane: {median_delay:.1f} min')

ax1.set_xlabel('Retard (minutes)', fontsize=12)
ax1.set_ylabel('Fréquence', fontsize=12)
ax1.set_title('Distribution des retards (vols retardés uniquement)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)

plt.tight_layout()
plt.savefig('graphique1_distribution_retards.png', dpi=150, bbox_inches='tight')
plt.close()

print("Graphique 1 enregistré: graphique1_distribution_retards.png")

print("""
+Justification du type de graphique:
+- Histogramme avec courbe de densité (KDE): idéal pour visualiser la distribution d'une variable continue
+- Permet de voir la forme de la distribution (asymétrie, pics)
+- Les lignes verticales montrent clairement la différence entre moyenne et médiane
+
+Observation: La distribution est asymétrique positive (étirée vers la droite), ce qui signifie
+que la plupart des retards sont modérés, mais il y a quelques retards très importants qui 
+tirent la moyenne vers le haut.
+""")


# Graphique 2 — Retard selon la saison et le statut
print("\n--- Graphique 2: Retard selon la saison et le statut ---")

fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

# Sous-graphe 1: Retard moyen par saison
season_order = ['Hiver', 'Printemps', 'Été', 'Automne']
delay_by_season = df.groupby('season')['delay_minutes'].mean().reindex(season_order)

colors_season = sns.color_palette("coolwarm", 4)
bars1 = axes[0].bar(season_order, delay_by_season.values, color=colors_season, edgecolor='black')
axes[0].set_xlabel('Saison', fontsize=12)
axes[0].set_ylabel('Retard moyen (minutes)', fontsize=12)
axes[0].set_title('Retard moyen par saison', fontsize=14, fontweight='bold')

# Ajouter les valeurs sur les barres
for bar, val in zip(bars1, delay_by_season.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 f'{val:.1f}', ha='center', va='bottom', fontsize=10)

# Sous-graphe 2: Répartition des statuts par saison
status_by_season = pd.crosstab(df['season'], df['status'], normalize='index') * 100
status_by_season = status_by_season.reindex(season_order)

status_order = ['À l\'heure', 'En avance', 'Retardé', 'Annulé']
status_by_season = status_by_season[status_order]
status_by_season.plot(kind='bar', stacked=True, ax=axes[1], 
                      color=['green', 'blue', 'orange', 'red'], edgecolor='black')

axes[1].set_xlabel('Saison', fontsize=12)
axes[1].set_ylabel('Pourcentage (%)', fontsize=12)
axes[1].set_title('Répartition des statuts par saison', fontsize=14, fontweight='bold')
axes[1].legend(title='Statut', bbox_to_anchor=(1.02, 1), loc='upper left')
axes[1].set_xticklabels(season_order, rotation=0)

plt.tight_layout()
plt.savefig('graphique2_saison_statut.png', dpi=150, bbox_inches='tight')
plt.close()

print("Graphique 2 enregistré: graphique2_saison_statut.png")

print("""
+Justification du type de graphique:
+- Graphique à barres (gauche): permet de comparer les retards moyens entre saisons
+- Graphique à barres empilées (droite): montre la composition des statuts pour chaque saison
+
+Conclusion: La saison du Printemps et Été pose le plus de problèmes à AirNova avec les retards 
les plus élevés. L'été a également le taux d'annulation le plus élevé (4.67%).
""")


# Graphique 3 — Relation prix / taux de remplissage
print("\n--- Graphique 3: Relation prix / taux de remplissage ---")

fig3, ax3 = plt.subplots(figsize=(10, 7))

# Scatter plot avec coloration par classe de cabine
cabin_colors = {'Économique': '#3498db', 'Affaires': '#e74c3c', 'Première': '#f39c12'}
cabin_markers = {'Économique': 'o', 'Affaires': 's', 'Première': '^'}

for cabin in df['cabin_class'].unique():
    subset = df[df['cabin_class'] == cabin]
    ax3.scatter(subset['load_factor_pct'], subset['ticket_price_eur'],
                c=cabin_colors.get(cabin, 'gray'), 
                marker=cabin_markers.get(cabin, 'o'),
                s=80, alpha=0.7, edgecolor='black', label=cabin)

ax3.set_xlabel('Taux de remplissage (%)', fontsize=12)
ax3.set_ylabel('Prix du billet (€)', fontsize=12)
ax3.set_title('Relation entre prix et taux de remplissage par classe de cabine', fontsize=14, fontweight='bold')
ax3.legend(title='Classe')

# Ajouter une ligne de tendance générale
z = np.polyfit(df['load_factor_pct'], df['ticket_price_eur'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['load_factor_pct'].min(), df['load_factor_pct'].max(), 100)
ax3.plot(x_line, p(x_line), "k--", alpha=0.5, label='Tendance générale')

plt.tight_layout()
plt.savefig('graphique3_prix_remplissage.png', dpi=150, bbox_inches='tight')
plt.close()

print("Graphique 3 enregistré: graphique3_prix_remplissage.png")

print("""
+Justification du type de graphique:
+- Nuage de points (scatter plot): idéal pour visualiser la relation entre deux variables continues
+- Chaque point représente un vol, coloré par classe de cabine
+- Permet de voir les patterns et clusters
+
+Conclusion sur la stratégie tarifaire d'AirNova:
+- Les billets en classe Affaires et Première sont généralement plus chers, quelle que soit la demande
+- Il n'y a pas de relation linéaire forte entre le prix et le taux de remplissage
+- La stratégie tarifaire semble être basée principalement sur la classe de cabine plutôt que sur 
+  l'anticipation de la demande (remplissage)
+- Les vols très remplis peuvent avoir des prix bas (Économique) ou élevés (Affaires)
+""")

# Graphique 4 — Carte thermique de corrélation
print("\n--- Graphique 4: Carte thermique de corrélation ---")

fig4, ax4 = plt.subplots(figsize=(14, 10))

# Matrice de corrélation
corr_matrix = df[numeric_cols].corr()

# Heatmap annotée
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax4, 
            annot_kws={'size': 9}, vmin=-1, vmax=1)

ax4.set_title('Carte thermique de corrélation des variables numériques', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('graphique4_heatmap_correlation.png', dpi=150, bbox_inches='tight')
plt.close()

print("Graphique 4 enregistré: graphique4_heatmap_correlation.png")

print("""
+Justification du type de graphique:
+- Heatmap (carte thermique): idéal pour visualiser une matrice de corrélation
+- Les couleurs permettent d'identifier rapidement les zones de forte corrélation
+- Les annotations numériques facilitent la lecture des valeurs
+
+Zones de forte corrélation positive:
+- flight_duration_min / fuel_consumed_kg (0.93): Plus le vol est long, plus il consomme de carburant
+- flight_duration_min / distance_km (0.91): La durée est proportionnelle à la distance
+- passengers / seat_capacity (0.67): Les gros appareils ont plus de passengers
+
+Zones de forte corrélation négative:
+- passenger_density / distance_km (-0.76): Les vols courts ont plus de passengers par km
+- passenger_density / flight_duration_min (-0.73): Même logique pour la durée
+""")


# Graphique 5 — Graphique libre au choix
# Choice: Retard moyen par heure de départ
print("\n--- Graphique 5: Retard moyen par heure de départ ---")

fig5, ax5 = plt.subplots(figsize=(12, 6))

# Calcul du retard moyen par heure de départ
delay_by_hour = df.groupby('departure_hour')['delay_minutes'].agg(['mean', 'count']).reset_index()

# Barres avec couleur basée sur l'intensité du retard
colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(delay_by_hour)))
bars = ax5.bar(delay_by_hour['departure_hour'], delay_by_hour['mean'], color=colors, edgecolor='black')

# Ajouter une ligne pour le retard moyen global
global_mean = df['delay_minutes'].mean()
ax5.axhline(global_mean, color='red', linestyle='--', linewidth=2, label=f'Retard moyen global: {global_mean:.1f} min')

# Ajouter les valeurs sur les barres
for bar, val in zip(bars, delay_by_hour['mean']):
    if val > 0:
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 f'{val:.1f}', ha='center', va='bottom', fontsize=8)

ax5.set_xlabel('Heure de départ', fontsize=12)
ax5.set_ylabel('Retard moyen (minutes)', fontsize=12)
ax5.set_title('Retard moyen par heure de départ', fontsize=14, fontweight='bold')
ax5.set_xticks(range(0, 24))
ax5.legend(fontsize=10)

# Mettre en évidence les heures de pointe
for i, hour in enumerate(delay_by_hour['departure_hour']):
    if hour in [7, 8, 9, 17, 18, 19]:
        ax5.get_xticklabels()[i].set_fontweight('bold')
        ax5.get_xticklabels()[i].set_color('red')
plt.tight_layout()
plt.savefig('graphique5_retard_heure.png', dpi=150, bbox_inches='tight')
plt.close()

print("Graphique 5 enregistré: graphique5_retard_heure.png")

print("""
Justification du type de graphique:
+- Graphique à barres: idéal pour comparer des valeurs par catégorie (les 24 heures)
+- Les couleurs permettent d'identifier visuellement les heures à risque
+- Ligne horizontale de référence pour le retard moyen global
+
+Question à laquelle ce graphique répond:
+- À quelle heure de la journée les retards sont-ils les plus élevés ?
+- Les heures de pointe (7-9h et 17-19h) sont-elles les plus problématiques ?
+
+Conclusion: Ce graphique apporte une information nouvelle en montrant que les retards 
+ne sont pas seulement liés aux heures de pointe classiques. Certaines heures de la nuit 
+(0h, 3h) ou du matin (5h, 6h) peuvent avoir des retards moyens très élevés, probablement 
+liés à des problèmes de correspondances ou de rotation des équipages.
+""")

 