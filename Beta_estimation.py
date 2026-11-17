import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme.expressions import Beta, Variable
from biogeme.models import loglogit

# 1. Importer le design que vous avez sauvegardé
df_questions = pd.read_csv("mon_design_pretest.csv")

# 2. Importer les réponses de votre sondage
file_path = "pretest_results.csv"
df_survey = pd.read_csv(file_path)

# 3. Transformation en format long pour Biogeme
observations = []
for index, row in df_survey.iterrows():
    # Vérifier si la personne a répondu à la première question
    if pd.isna(row.get("Situation 1 sur 13 : quel pull choisiriez-vous ?")):
        continue
        
    for q in range(1, 13): 
        nom_colonne = f"Situation {q} sur 13 : quel pull choisiriez-vous ?"
        if nom_colonne not in row or pd.isna(row[nom_colonne]):
            continue
            
        choix_texte = row[nom_colonne]
        choix_code = 1 if choix_texte == "Seconde Main" else 2
        
        # On va chercher les attributs directement dans le df_questions importé
        design_q = df_questions[df_questions['Question'] == q].iloc[0]
        
        obs = {
            'Respondent_ID': index + 1,  # Utilise l'index de la boucle pour créer un ID 1, 2, 3... purement numérique
            'CHOIX': choix_code,
            'PRIX_SM': design_q['Prix_SM'],
            'QUALITE_SM': design_q['Qualite_SM'],
            'UNICITE_SM': design_q['Unicite_SM'],
            'RETOURS_SM': design_q['Retours_SM'],
            'PRIX_NEUF': design_q['Prix_neuf'],
            'UNICITE_NEUF': design_q['Unicite_neuf'],
            'RETOURS_NEUF': design_q['Retours_neuf']
        }
        observations.append(obs)

df_biogeme = pd.DataFrame(observations)

# Aperçu pour vérifier que tout est bien fusionné
print(df_biogeme.head())

# 3. Création de la base de données Biogeme
database = db.Database("PreTest_SCM", df_biogeme)

# Définition des variables
CHOIX = Variable('CHOIX')
PRIX_SM = Variable('PRIX_SM')
QUALITE_SM = Variable('QUALITE_SM')
UNICITE_SM = Variable('UNICITE_SM')
RETOURS_SM = Variable('RETOURS_SM')
PRIX_NEUF = Variable('PRIX_NEUF')
UNICITE_NEUF = Variable('UNICITE_NEUF')
RETOURS_NEUF = Variable('RETOURS_NEUF')

# 4. Définition des Betas à estimer (Priors de départ à 0)
asc_sm = Beta('ASC_SM', 0, None, None, 0)
beta_prix = Beta('B_PRIX', 0, None, None, 0)
beta_qualite = Beta('B_QUALITE', 0, None, None, 0)
beta_unicite = Beta('B_UNICITE', 0, None, None, 0)
beta_retours_sm = Beta('B_RETOURS_SM', 0, None, None, 0)
beta_retours_neuf = Beta('B_RETOURS_NEUF', 0, None, None, 0)

# 5. Fonctions d'utilité (Basées sur votre modèle théorique)
V1 = (asc_sm + 
      beta_prix * PRIX_SM + 
      beta_qualite * (QUALITE_SM == 2) + 
      beta_unicite * (UNICITE_SM == 2) + 
      beta_retours_sm * (RETOURS_SM == 2))

V2 = (beta_prix * PRIX_NEUF + 
      beta_unicite * (UNICITE_NEUF == 2) + 
      beta_retours_neuf * (RETOURS_NEUF == 2))

V = {1: V1, 2: V2}
av = {1: 1, 2: 1} # Les deux alternatives sont toujours disponibles

# 6. Lancement de l'estimation
logprob = loglogit(V, av, CHOIX)
biogeme = bio.BIOGEME(database, logprob)
biogeme.modelName = "Modele_Pre_Test"
results = biogeme.estimate()

# Extraction des résultats
betas_df = results.get_estimated_parameters()
pd.set_option('display.max_columns', None)
print(betas_df)
# Sauvegarder les priors calculés pour le prochain notebook
df_a_sauvegarder = betas_df["Estimated parameters"]
df_a_sauvegarder.to_csv("mes_priors_estimes.csv", index=False)