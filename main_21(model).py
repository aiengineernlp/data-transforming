""" Préparation des données pour le Machine Learning"""
import pandas as pd

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder

print("📁 1. Importation des données\n")
reviews21 = pd.read_csv("data/reviews_clients_vetement2.csv")
# convertir en reviews21 en df avec DataFrame de pandas
df = pd.DataFrame(reviews21)
print("les premieres lignes du dataset: \n", df.head(5))
print("la liste des colonnes:\n", df.columns)
print("print les types de donnees: \n", df.dtypes)
print("les informations generales sur les donnees:\n", df.info())
print("\n\n\n")
print("🧭2. Exploration des données  (les etiquettes/valeur unique/categories, type d'encodage et raison) \n")

# print("values_count():\n","Type encodage:\n","Raison: ","\n\n")
print(f"values_count():{df['clothing_id'].value_counts()}\n","Type encodage: On scale juste../ou alors on fait la copie\n","Raison: La variable categorielle est l'ensemble des valeurs discrettes continue discrette (le chiffres positifs)","\n\n")
print(f"values_count():{df['age'].value_counts()}\n","Type encodage: On scale juste\n","Raison: la variable categorielle , ce sont des valeurs  continues discrettes (chiffres positis)","\n\n")

print(f"values_count():{df['Title'].value_counts()}\n","Type encodage:Text vectorization de nlp\n","Raison: la variable `categorielle est l'ensembles de texte libre semantique (qui a du sens) ecris pas les utilisateurs alors il faut passewr par le NLP pour l'encodage ","\n\n")

print(f"values_count():{df['Review Text'].value_counts()}\n","Type encodage:Text vectorization de NLP\n","Raison: la Variable Categorielle est l'ensemble de texte semantique (texte libre ayant du sens entrer par les utilisateurs) ","\n\n")

print(f"values_count():{df['Rating'].value_counts()}\n","Type encodage:ordinal ou via .map grace au dictionnaire\n","Raison: Car la variable categorielle a une structure ordonnee/ hierarchiser genre (niveau 1, niveau 2, niveau 3,niveau 4,niveau 5.) tres visible ","\n\n")
print(f"values_count():{df['Recommended IND'].value_counts()}\n","Type encodage:Binaire/LabelEncoder/juste une copie/ \n","Raison:  car juste deux categories(etiquettes/valeur unique) a ne pas pas confondre avec BinaryEncoder MAIS ON NE VA PLUS ENCODER CAR C'EST DEJA EN CHIFFRE/ ON VA DONC FAIRE JUSTE UNE COPIE OU  OU ALORS POUR CE CAS FAIRE UN LABEL ENCODER CAR ON SE RAPPEL QUE UN TYPE NOMINAL","\n\n")
print(f"values_count():{df['Positive Feedback Count'].value_counts()}\n","Type encodage: On scale juste ou alors on fait la copie\n","Raison: La VC est un ensemble de valeurs continue discrettes(les chiffres positifs) ","\n\n")
print(f"values_count():{df['Division Name'].value_counts()}\n","Type encodage: One hot\n","Raison: la Variable categorielle compte moins de 10 categories ","\n\n")
print(f"values_count():{df['Department Name'].value_counts()}\n","Type encodage: One hot\n","Raison: la Variable categorielle compte moins de 10 categories ","\n\n")
print(f"values_count():{df['Class Name'].value_counts()}\n","Type encodage:  One hot\n","Raison: la VC compte moins de 10 categories ","\n\n")
print("CAS DE review_date\n\n\n\n\n\n\n\n")
print("convertir 'review_date' de object a datetime\n ")
df['review_date_to_date_time'] =pd.to_datetime(df['review_date'])
print("\n\n\n")
print("🔁 3. Encodage des variables catégorielles\n")
print("CAS DE review_date_to_date_time")
print("Jour random\n")
df['jour_random'] = df['review_date_to_date_time'].dt.day
print("le resultat est :\n ",df['jour_random'] )
print("\n\n")

print("Jour de la semaine\n")
df['dayofweek'] = df['review_date_to_date_time'].dt.dayofweek
print("le resultat est :\n ",df['dayofweek'] )
print("\n\n")

print("Le week - end\n")
df['is_weekend'] = df['dayofweek'].apply(lambda x:1 if x>=5 else 0)  # explique moi bien ici
print("le resultat est :\n ",df['is_weekend'] )
print("\n\n")

print("pour le mois \n")
df['months'] = df['review_date_to_date_time'].dt.month
print("le resultat est :\n ",df['months'] )
print("\n\n")


print("trimester ~(quarter) \n")
df['quarter'] = df['review_date_to_date_time'].dt.quarter
print("le resultat est :\n ",df['quarter'] )
print("\n\n")

print("annee \n")
df['year'] = df['review_date_to_date_time'].dt.year
print("le resultat est :\n ",df['year'] )
print("\n\n")

print("\n\n\n")
print("🔁 3. Encodage des variables catégorielles\n")
print("CAS DE clothing_id")
sc  = StandardScaler()
df['clothing_id_encoder'] = sc.fit_transform(df[['clothing_id']])
print("le resultat est :\n", df['clothing_id_encoder'])
print("\n\n\n")
print("CAS DE age")
df['age_encoder'] = sc.fit_transform(df[['age']])
print("le resultat est :\n", df['age_encoder'])

print("\n\n\n")
print("CAS DE Rating")
#--->> Methode 1: ENCODAGE AVEC .map
'''Creation dun dictionnaire'''
rating_21_dict = {
	"Loved it" : 5,
	"Liked it" : 4,
	"Was okay" : 3,
	"Not great" : 2,
	"Hated it" : 1
}
df['Rating_encoder_map'] = df['Rating'].map(rating_21_dict)
print("le resultat est :\n", df['Rating_encoder_map'])

#--->> Methode 2: ENCODAGE AVEC OrdinalEncoder
#--->> Retirer les espaces en fin et en debut des chaines de caracteres
reviews21['Rating'] = reviews21['Rating'].astype(str).str.strip()
# Vérification des valeurs uniques restantes
print("Valeurs uniques après nettoyage :", df['Rating'].unique())
# Définir les valeurs connues et autorisées
valid_categories = ["Loved it", "Liked it", "Was okay", "Not great", "Hated it"]
# Filtrer les valeurs valides pour éviter l'erreur avec OrdinalEncoder
df_valid = df[df['Rating'].isin(valid_categories)].copy()
# Reshape (remodelisation de la variable)
rating_reshaped = df_valid['Rating'].values.reshape(-1, 1)

# Encoder avec ordre défini
encoder = OrdinalEncoder(categories=[valid_categories])
df_valid['Rating_encoder'] = encoder.fit_transform(rating_reshaped)
# Bonus : Ajouter une ligne pour détecter les valeurs inconnues
print("Valeurs non reconnues :", df[~df['Rating'].isin(valid_categories)]['Rating'].unique())

# Affichage
print(df_valid[['Rating', 'Rating_encoder']])

print("\n\n\n")
print("CAS DE Recommended IND --->> binaire(1 ou 2)")
le = LabelEncoder()
df['Recommended_IND_encoder'] = le.fit_transform(df['Recommended IND'])
print("le resultat est :\n", df['Recommended_IND_encoder'])





