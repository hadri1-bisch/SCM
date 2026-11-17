
import numpy as np
import pandas as pd
from choicedesign.expressions import Attribute, Parameter, ASC
from choicedesign.design import EffDesign
from itertools import product

# ==================================================
# 1. RÉCUPÉRATION DE VOS PRIORS DU PRÉ-TEST
# (Les valeurs extraites de votre modèle Biogeme)
# ==================================================
betas_df = pd.read_csv("mes_priors_estimes.csv")
# On s'assure que le DataFrame des betas de Biogeme a la colonne "Name" en index
if "Name" in betas_df.columns:
    betas_df = betas_df.set_index("Name")

# Extraction des valeurs. Vous avez choisi de garder TOUS les priors (significatifs ou non).
# Note : on utilise la colonne "Value" de votre sortie Biogeme.
val_asc_sm = betas_df.loc["ASC_SM", "Value"]
val_b_prix = betas_df.loc["B_PRIX", "Value"]
val_b_qualite = betas_df.loc["B_QUALITE", "Value"]
val_b_unicite = betas_df.loc["B_UNICITE", "Value"]
val_b_retours_sm = betas_df.loc["B_RETOURS_SM", "Value"]
val_b_retours_neuf = betas_df.loc["B_RETOURS_NEUF", "Value"]

print(f"Priors injectés -> Prix: {val_b_prix:.4f}, Qualité: {val_b_qualite:.4f}, "
      f"Unicité: {val_b_unicite:.4f}, Retours SM: {val_b_retours_sm:.4f}, "
      f"Retours Neuf: {val_b_retours_neuf:.4f}, ASC SM: {val_asc_sm:.4f}")


# ==================================================
# 2. REDÉFINITION DES ATTRIBUTS ET PARAMÈTRES POUR LE DCE FINAL
# ==================================================
prix_sm_niveaux = [20, 40]
qualite_sm_niveaux = [1, 2]
unicite_sm_niveaux = [1, 2]
garantie_sm_niveaux = [1, 2]

prix_neuf_niveaux = [40, 60, 80] # Corrigé à [40, 60, 80] comme dans votre version finale
unicite_neuf_niveaux = [1, 2]
garantie_neuf_niveaux = [1, 2]

prix_sm = Attribute("Prix_SM", prix_sm_niveaux)
qualite_sm = Attribute("Qualite_SM", qualite_sm_niveaux)
unicite_sm = Attribute("Unicite_SM", unicite_sm_niveaux)
retours_sm = Attribute("Retours_SM", garantie_sm_niveaux)

prix_neuf = Attribute("Prix_neuf", prix_neuf_niveaux)
unicite_neuf = Attribute("Unicite_neuf", unicite_neuf_niveaux)
retours_neuf = Attribute("Retours_neuf", garantie_neuf_niveaux)

# Définition des Parameter avec les vrais priors extraits
beta_prix = Parameter("beta_prix", val_b_prix)
beta_unicite = Parameter("beta_unicite", val_b_unicite)
beta_qualite = Parameter("beta_qualite", val_b_qualite)
beta_retours_sm = Parameter("beta_retours_sm", val_b_retours_sm)
beta_retours_neuf = Parameter("beta_retours_neuf", val_b_retours_neuf)
asc_sm = ASC("asc_sm", val_asc_sm)

# Fonctions d'utilité avec les VRAIS priors
V_sm_final = (
    asc_sm
    + beta_prix * prix_sm
    + beta_qualite * (qualite_sm == 2)
    + beta_unicite * (unicite_sm == 2)
    + beta_retours_sm * (retours_sm == 2)
)

V_neuf_final = (
    beta_prix * prix_neuf
    + beta_unicite * (unicite_neuf == 2)
    + beta_retours_neuf * (retours_neuf == 2)
)

V_final = {1: V_sm_final, 2: V_neuf_final}


# ==================================================
# 3. CRÉATION DES CANDIDATS (Plan Factoriel Complet filtré)
# ==================================================
import itertools
profils_neuf = pd.DataFrame(
    list(itertools.product(prix_neuf_niveaux, unicite_neuf_niveaux, garantie_neuf_niveaux)),
    columns=['Prix', 'Unicité', 'Garantie']
)

profils_sm = pd.DataFrame(
    list(itertools.product(prix_sm_niveaux, qualite_sm_niveaux, unicite_sm_niveaux, garantie_sm_niveaux)),
    columns=['Prix', 'Qualité', 'Unicité', 'Garantie']
)

paires = []
for (idx_sm, profil_sm), (idx_neuf, profil_neuf) in product(profils_sm.iterrows(), profils_neuf.iterrows()):
    paires.append({
        "Prix_SM": profil_sm["Prix"],
        "Qualite_SM": profil_sm["Qualité"],
        "Unicite_SM": profil_sm["Unicité"],
        "Retours_SM": profil_sm["Garantie"],
        "Prix_neuf": profil_neuf["Prix"],
        "Unicite_neuf": profil_neuf["Unicité"],
        "Retours_neuf": profil_neuf["Garantie"]
    })

df_paires_candidates = pd.DataFrame(paires)

def est_evident(row):
    sm_mieux = (row["Prix_SM"] <= row["Prix_neuf"]
                and row["Unicite_SM"] >= row["Unicite_neuf"]
                and row["Retours_SM"] >= row["Retours_neuf"])
    neuf_mieux = (row["Prix_neuf"] <= row["Prix_SM"]
                  and row["Unicite_neuf"] >= row["Unicite_SM"]
                  and row["Retours_neuf"] >= row["Retours_SM"])
    return sm_mieux or neuf_mieux

candidates = (
    df_paires_candidates[~df_paires_candidates.apply(est_evident, axis=1)]
    .reset_index(drop=True)
)


# ==================================================
# 4. RECHERCHE LOCALE D-EFFICIENT (Algorithme d'optimisation)
# ==================================================
colonnes_modele = [
    "Prix_SM", "Qualite_SM", "Unicite_SM", "Retours_SM",
    "Prix_neuf", "Unicite_neuf", "Retours_neuf"
]

design = EffDesign(
    X=[prix_sm, qualite_sm, unicite_sm, retours_sm, prix_neuf, unicite_neuf, retours_neuf],
    ncs=12
)

def evaluer_selection(indices):
    plan = candidates.iloc[indices][colonnes_modele].copy()
    nb_standard = (plan["Qualite_SM"] == 1).sum()
    penalite = 0.0
    if nb_standard < 5:
        penalite = (5 - nb_standard) * 10.0

    plan.insert(0, "CS", range(1, len(plan) + 1))
    
    # ICI ON UTILISE V_final AVEC LES VRAIS PRIORS
    d_error, equilibre = design.evaluate(
        plan,
        V=V_final, 
        criterion="d"
    )
    return float(d_error) + penalite, float(equilibre)

rng = np.random.default_rng(42)
meilleur_plan = None
meilleur_d_error = np.inf
meilleur_equilibre = None

nombre_depart = 5
iterations_max = 25
taille_echantillon = 70

for depart in range(nombre_depart):
    while True:
        selection = list(rng.choice(len(candidates), size=12, replace=False))
        if (candidates.iloc[selection]["Qualite_SM"] == 1).sum() >= 5:
            break

    d_error, equilibre = evaluer_selection(selection)

    for iteration in range(iterations_max):
        indices_selectionnes = set(selection)
        candidats_restants = [i for i in range(len(candidates)) if i not in indices_selectionnes]
        candidats_testes = rng.choice(
            candidats_restants,
            size=min(taille_echantillon, len(candidats_restants)),
            replace=False
        )

        meilleur_echange = None
        meilleur_d_local = d_error
        meilleur_eq_local = equilibre

        for position in range(12):
            for candidate_idx in candidats_testes:
                essai = selection.copy()
                essai[position] = candidate_idx
                d_essai, eq_essai = evaluer_selection(essai)

                if np.isfinite(d_essai) and d_essai < meilleur_d_local - 1e-10:
                    meilleur_d_local = d_essai
                    meilleur_eq_local = eq_essai
                    meilleur_echange = essai

        if meilleur_echange is None:
            break

        selection = meilleur_echange
        d_error = meilleur_d_local
        equilibre = meilleur_eq_local

    print(f"Départ {depart + 1}/{nombre_depart} terminé — D-error : {d_error:.4f}")

    if d_error < meilleur_d_error:
        meilleur_d_error = d_error
        meilleur_equilibre = equilibre
        meilleur_plan = selection.copy()

# ==================================================
# 5. AFFICHAGE DES 12 QUESTIONS FINALES
# ==================================================
df_questions_finales = candidates.iloc[meilleur_plan].copy()
df_questions_finales = df_questions_finales.reset_index(drop=True)
df_questions_finales.insert(0, "Question", range(1, len(df_questions_finales) + 1))

print("\n--- DESIGN FINAL OPTIMISÉ ---")
print("Nombre de questions :", len(df_questions_finales))
print("D-error final :", meilleur_d_error)
print("Équilibre des utilités (%) :", meilleur_equilibre)

pd.set_option('display.max_columns', None)
print(df_questions_finales)


# Dictionnaires de correspondance code -> libellé
labels_qualite = {1: "Bon état, avec traces d'ususre visible", 2: "Comme neuf"}
labels_unicite = {1: "Pièce mainstream", 2: "Trouvaille unique"}
labels_garantie = {1: "Sans retour", 2: "Retours 30j"}

df_questions_lisible = df_questions_finales.copy()

df_questions_lisible["Qualite_SM"] = df_questions_lisible["Qualite_SM"].replace(labels_qualite)
df_questions_lisible["Unicite_SM"] = df_questions_lisible["Unicite_SM"].replace(labels_unicite)
df_questions_lisible["Retours_SM"] = df_questions_lisible["Retours_SM"].replace(labels_garantie)
df_questions_lisible["Unicite_neuf"] = df_questions_lisible["Unicite_neuf"].replace(labels_unicite)
df_questions_lisible["Retours_neuf"] = df_questions_lisible["Retours_neuf"].replace(labels_garantie)

print(df_questions_lisible)

# ==================================================
# 6. IMPRESSIONS POUR SURVEY
# ==================================================

from IPython.display import display, HTML

# Parcourir les 12 questions
for idx, row in df_questions_lisible.iterrows():
    num_q = int(row['Question'])

    # Données pour SM et Neuf
    col_sm = [
        f"{row['Prix_SM']} CHF",
        row.get('Qualite_SM'),
        row.get('Unicite_SM'),
        row.get('Retours_SM')
    ]

    col_neuf = [
        f"{row['Prix_neuf']} CHF",
        row.get('Qualite_neuf', 'Neuf'),
        row.get('Unicite_neuf'),
        row.get('Retours_neuf')
    ]

    # Alterner le sens gauche/droite pour l'effet A / B
    inverser = (num_q % 2 == 0)

    if inverser:
        data_paire = {"Neuf": col_neuf, "Seconde Main": col_sm}
    else:
        data_paire = {"Seconde Main": col_sm, "Neuf": col_neuf}

    df_q = pd.DataFrame(data_paire, index=['Prix', 'Qualité', 'Unicité', 'Retours'])

    # Application du style : fond blanc pur, lignes horizontales pour chaque attribut, et trait vertical très net
    styled_df = df_q.style \
        .set_caption(f"<b>Question {num_q} / 12</b>") \
        .set_table_styles([
            {'selector': 'caption', 'props': [('font-size', '16px'), ('font-weight', 'bold'), ('text-align', 'left'), ('margin-bottom', '10px'), ('color', '#2c3e50')]},
            {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('margin-bottom', '30px'), ('width', '100%'), ('max-width', '600px'), ('font-family', 'sans-serif'), ('background-color', '#ffffff'), ('box-shadow', '0 4px 6px rgba(0,0,0,0.05)')]},
            {'selector': 'th', 'props': [('background-color', '#ffffff'), ('color', '#1e293b'), ('font-weight', '700'), ('font-size', '14px'), ('padding', '10px 14px'), ('text-align', 'center'), ('border-bottom', '2px solid #cbd5e1')]},
            {'selector': 'td', 'props': [('background-color', '#ffffff'), ('color', '#1e293b'), ('padding', '10px 14px'), ('border-bottom', '1px solid #e2e8f0'), ('text-align', 'center'), ('font-size', '13px')]},

            # Colonne des attributs (gauche) : fond blanc et lignes horizontales conservées, avec un trait vertical très net à droite
            {'selector': 'th.row_heading', 'props': [('background-color', '#ffffff'), ('color', '#0f172a'), ('font-weight', '600'), ('width', '100px'), ('text-align', 'left'), ('border-right', '3px solid #1e293b'), ('border-bottom', '2px solid #cbd5e1')]},
            {'selector': 'td.row_heading', 'props': [('background-color', '#ffffff'), ('color', '#0f172a'), ('font-weight', '600'), ('text-align', 'left'), ('border-right', '3px solid #1e293b'), ('border-bottom', '1px solid #e2e8f0')]},

            # Fermer le bloc tout en haut à gauche
            {'selector': 'thead tr th:first-child', 'props': [('border-right', '3px solid #1e293b'), ('background-color', '#ffffff')]},

            # Ligne de séparation entre l'Alternative A et l'Alternative B (milieu)
            {'selector': 'td:nth-child(2), th:nth-child(2)', 'props': [('border-right', '2px solid #cbd5e1')]}
        ])

    display(styled_df)
    
    # Générer le code HTML du tableau
    html_content = styled_df.to_html()
    
    # Enregistrer chaque tableau dans un fichier HTML distinct
    filename = f"Question_Qualtrics_{num_q}.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)

print("Les 12 fichiers HTML (Question_Qualtrics_1.html, etc.) ont été créés dans votre dossier.")