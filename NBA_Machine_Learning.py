# Load libraries
import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import glob
import numpy as np
import datetime

#Load Dataset
dataframe_games = pd.read_csv("2012-18_teamBoxScore.csv")
dataframe_standings = pd.read_csv("2012-18_standings.csv")

#Looking at shape of data
#print(dataframe_games.shape)
#print(dataframe_standings.describe())
#dataframe_standings.plot(kind = "box", subplots = True, layout = (5, 5), sharex = False, sharey = False)
#dataframe_standings.hist()
#pyplot.show()

#get games as dataframe_games has 2 rows for each game - one for home, one for away
#away is first, home is second

#Data Cleaning
games_array = dataframe_games.values
standings_array = dataframe_standings.values

game_list = []

for i in range(6, len(games_array), 2):
    date = games_array[i][0]
    away_team = games_array[i][9]
    home_team = games_array[i+1][9]
    game_list.append([date, away_team, home_team])

# can only predict game from past results
# start from 2nd day of play, hence starting loop from 6

#get x_data

try:
    final_array = np.load("nba_data.npy")
    print(len(final_array))
except: 
    print("excepted")
    final_array = []

    for i in range(len(games_array)):
        date = games_array[i][0]
        team = games_array[i][9]

        date = datetime.datetime.strptime(date, "%d/%m/%Y").date()
        yesterday_date_object = date + datetime.timedelta(days=-1)
        day_before_yesterday_object = yesterday_date_object + + datetime.timedelta(days=-1)
        yesterday_date = yesterday_date_object.strftime("%d/%m/%Y")
        day_before_yesterday = day_before_yesterday_object.strftime("%d/%m/%Y")

        if yesterday_date[0] == "0":
            yesterday_date = yesterday_date[1:]

        if day_before_yesterday[0] == "0":
            day_before_yesterday = day_before_yesterday[1:]

        print(len(final_array), yesterday_date)
        
        """
        Standings
        0 : Date
        1 : Team
        2 : Rank
        4 : Games Won
        5 : Games Lost
        9 : Games Back
        10 : Pts For
        11 : Pts Against
        12 : Home Win
        14 : Away Win
        18 : Last 5
        19 : Last 10
        21 : Pts Score avg
        22 : Pts Allow avg

        Games Array
        15 : Days off
        16 : Team Pts

        Format
        [date, team, rank, gamesW, gamesL, gamesB, ptsFor, ptsAgainst, homeW, awayW, last5, last10, ptsScoreAvg, ptsAllowAvg, daysOff, teamPts]
        """
        for standing in standings_array:
            if standing[0] == yesterday_date and standing[1] == team:
                full_array = list(standing[[0, 1, 2, 4, 5, 9, 10, 11, 12, 14, 18, 19, 21, 22]]) + [games_array[i][15]] + [games_array[i][16]]
                final_array.append(full_array)
                break
            elif standing[0] == day_before_yesterday and standing[1] == team:
                full_array = list(standing[[0, 1, 2, 4, 5, 9, 10, 11, 12, 14, 18, 19, 21, 22]]) + [games_array[i][15]] + [games_array[i][16]]
                final_array.append(full_array)
                break

    np_array = np.array(final_array)
    np.save("nba_data.npy", np_array)


#get victor
# [0] for away victory [1] for home victory

y_data = []
for i in range(6, len(final_array), 2):
    away_points = final_array[i][15]
    home_points = final_array[i+1][15]

    if away_points > home_points:
        y_data.append(0)
    else:
        y_data.append(1)

x_data = []
for i in range(0, len(final_array) - 6, 2):
    array_t1 = list(map(float, final_array[i][2:15]))
    array_t2 = list(map(float, final_array[i+1][2:15]))

    x_data.append(array_t1 + array_t2)

print(x_data[1032])
print(final_array[2064], final_array[2065])


x_train, x_validation, y_train, y_validation = train_test_split(x_data, np.ravel(y_data,order='C'), test_size=0.2, random_state=1)

#Checking for right model
models = []
models.append(("LR", LogisticRegression(solver="liblinear", multi_class="ovr")))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier()))
models.append(("NB", GaussianNB()))
models.append(("SVC", SVC(gamma="auto")))


#evaluating each model
results = []
names = []

for name, model in models:
    kfold = StratifiedKFold(n_splits = 10, random_state = 1, shuffle = True)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

#pyplot.boxplot(results, labels = names)
#pyplot.title("Algorithm Comparison")
#pyplot.show()

model = LogisticRegression(solver="liblinear", multi_class="ovr")
model.fit(x_train, y_train)
predictions = model.predict(x_validation)

# Evaluate predictions
print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))











   


    
    








