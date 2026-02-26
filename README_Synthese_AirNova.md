# Synthèse et Recommandations - AirNova

## 21. Profil type du vol retardé

L'analyse des données AirNova révèle que le vol le plus susceptible d'être retardé présente certaines caractéristiques spécifiques. En nous appuyant sur les résultats de nos modèles et analyses, le profil type d'un vol retardé est généralement un vol de durée moyenne à longue ( flight_duration_min élevé), opérant sur des routes internationales avec une distance significative. Les vols pendant les heures de pointe (7-9h et 17-19h) montrent une propension plus élevée aux retards en cascade. Le taux de remplissage (load_factor_pct) joue également un rôle important : un avion très occupé peut subir des retards liés aux temps d'embarquement plus longs. Les vols du printemps et du lundi montrent également des tendances de retard plus marquées selon nos analyses bivariées.

---

## 22. Bilan des modèles

Nous avons entraîné et comparé trois modèles de classification : l'Arbre de Décision, le Random Forest et la Régression Logistique.

| Modèle | Accuracy | F1-Score | Recall |
|--------|----------|----------|--------|
| Arbre de Décision | 0.75 | 0.12 | 0.08 |
| Random Forest | 0.80 | 0.00 | 0.00 |
| Régression Logistique | 0.80 | 0.00 | 0.00 |

**Recommandation pour AirNova :** 
Nous recommandons l'**Arbre de Décision** malgré ses métriques apparemment modestes. En termes métier, ce modèle est le seul à avoir réussi à identifier correctement les vols retardés (Recall de 0.08, contre 0 pour les autres). Pour une compagnie aérienne, il est crucial de détecter les retards potentiels, même avec des faux positifs. Un modèle qui prédit toujours "non retardé" (comme Random Forest et Régression Logistique) est inutile opérationnellement. L'Arbre de Décision offre également une interprétabilité essentielle : les équipes opérationnelles peuvent comprendre facilement les règles de décision.

---

## 23. Recommandations opérationnelles

### Recommandation 1 : Optimiser la gestion des heures de pointe
Les résultats montrent que les heures de départ 7-9h et 17-19h sont critiques. AirNova devrait :
- Augmenter les ressources au sol pendant ces créneaux
- Prévoir des tampons de temps supplémentaires entre les vols
- Anticiper les correspondances avec des marges de sécurité

### Recommandation 2 : Focus sur les vols longs et internationaux
La distance_km et flight_duration_min sont des variables très importantes. Nous recommandons :
- Une attention particulière aux vols long-courriers
- Des procédures accélérées pour les vols internationaux à forte demande
- Coordination renforcée avec les contrôleurs aériens sur les routes congestionnées

### Recommandation 3 : Gestion proactive du taux de remplissage
Le load_factor_pct influence significativement les retards. AirNova devrait :
- Limiter le taux de remplissage à 85% sur les routes à risque
- Mettre en place des procédures d'embarquement prioritaires
- Anticiper les retards en cascade avec des messages proactifs aux passagers

---

## 24. Limites de l'analyse

### Limite 1 : Taille réduite du dataset
Notre dataset ne contient que 300 vols, ce qui est insuffisant pour un modèle robuste. Avec plus de données (plusieurs milliers de vols), les modèles pourraient apprendre des patterns plus fiables et généraliser mieux.

### Limite 2 : Variables manquantes
Nous n'avons pas accès à des données cruciales comme :
- Les conditions météorologiques exactes
- L'état du trafic aérien
- Les informations sur les équipages (rotation, fatigue)
- Les historique de maintenance des avions
- Les retards en cascade des vols précédents

**Solution proposée :** Avec plus de ressources, AirNova pourrait intégrer un data lake complet avec des données météorologiques en temps réel, le statut des aéroports, et l'historique des équipements pour construire un modèle prédictif plus puissant.

---

