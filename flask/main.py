
# Librairies

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import ensemble, linear_model, tree

# import dataset
header = ["NbLikePage", "NbPersLieu", "NbInteractionQuot","Categorie", "5", "6","7", "8", "9","10", "11", "12","13", "14",
               "15","16", "17", "18","19", "20", "21","22", "23", "24","25", "26", "27","28", "29",
               "NbComAvantBD","NbCom24hAvantBD", "NbCom48hAvantBD", "NbCom24hApresPubli","DiffCom24h48h", "BaseDate",
               "NbCaracterePost","NbPartageAvantBD", "PostSponso", "Hhours","IfSundayPost", "IfMondayPost", "IfTuesdayPost",
               "IfWednesdayPost", "IfThursdayPost", "IfFridayPost","IfSaturdayPost","IfSundayBD", "IfMondayBD",
               "IfTuesdayBD","IfWednesdayBD", "IfThursdayBD", "IfFridayBD","IfSaturdayBD", "NbCom"]
#header = range(1,55)
train_dataset = [pd.read_csv("./Dataset/Training/Features_Variant_{}.csv".format(i), names = header, index_col=False) for i in range(1,6)]
train_dataset = pd.concat(train_dataset, ignore_index=True)
test_dataset = pd.read_csv("./Dataset/Testing/Features_TestSet.csv", names = header, index_col=False)

# Création Variable
conditions = [
    (train_dataset['IfMondayPost'] == 1),
    (train_dataset['IfTuesdayPost'] == 1),
    (train_dataset['IfWednesdayPost'] == 1),
    (train_dataset['IfThursdayPost'] == 1),
    (train_dataset['IfFridayPost'] == 1),
    (train_dataset['IfSaturdayPost'] == 1),
    (train_dataset['IfSundayPost'] == 1),
]

# create a list of the values we want to assign for each condition
values = [0, 1, 2, 3, 4, 5, 6]
# values = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi", "Dimanche"]
# create a new column and use np.select to assign values to it using our lists as arguments
train_dataset['JourSemaine'] = np.select(conditions, values)

# Cleaning dataset

# Delete all posts that were posted less than 24hours ago
train_dataset.drop(train_dataset[(train_dataset["NbCom48hAvantBD"] == 0) & (
            train_dataset["NbCom24hAvantBD"] == train_dataset["NbCom24hApresPubli"])].index, inplace=True)

# Only keep hhours of 24
train_dataset.drop(train_dataset[(train_dataset["Hhours"] != 24)].index, inplace=True)

# reset index
train_dataset.reset_index(drop=True)



flask_dataset = train_dataset.sample(10000)
flask_dataset = flask_dataset.drop(["5", "6", "7", "8", "9", "10", "11", "12", "13", "14",
                                    "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28",
                                    "29", "BaseDate", "IfSundayPost", "IfMondayPost", "IfTuesdayPost",
                                    "IfWednesdayPost", "IfThursdayPost", "IfFridayPost", "IfSaturdayPost", "IfSundayBD",
                                    "IfMondayBD",
                                    "IfTuesdayBD", "IfWednesdayBD", "IfThursdayBD", "IfFridayBD", "IfSaturdayBD"],
                                   axis=1)

y = flask_dataset["NbCom"]
x = flask_dataset.drop(["NbCom"], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

scaler = StandardScaler()
scaler.fit(x_train)  # il ne faut fiter que sur les données d entrainement
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print("Fitting model...")

boosting = ensemble.GradientBoostingClassifier(n_estimators=200, max_depth=1, learning_rate=.1)
boosting.fit(x_train, y_train)
print("Done fitting")


# Saving model to disk
pickle.dump(boosting, open('model.pkl', 'wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))

print(model.predict([[2, 9, 6, 3, 4, 65, 34, 12, 32, 45, 1, 3, 23, 34]]))

