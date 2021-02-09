import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

training_dataset = pd.read_csv("data/train.csv")
del training_dataset['Id']
del training_dataset['fnlwgt']
del training_dataset['education']
del training_dataset['native-country']
del training_dataset['race']
del training_dataset['relationship']
del training_dataset['marital-status']

# Encode some variables
le = OrdinalEncoder()
le.fit(training_dataset.iloc[:, [1, 3, 4]])
d = le.transform(training_dataset.iloc[:, [1, 3, 4]])
df1 = pd.DataFrame(d)

training_dataset['workclass'] = df1[0]
training_dataset['occupation'] = df1[1]
training_dataset['sex'] = df1[2]


# Categorize other variables
def categorize_age(age):
    if age <= 25:
        return 1
    elif 25 < age <= 45:
        return 2
    elif 45 < age <= 65:
        return 3
    elif age > 65:
        return 4


def categorize_gain(gain):
    if gain <= 3464:
        return 1
    elif 3464 < gain <= 14080:
        return 2
    elif gain > 14080:
        return 3


def categorize_loss(loss):
    if loss <= 1672:
        return 1
    elif 1672 < loss <= 1977:
        return 2
    elif loss > 1977:
        return 3


def categorize_hours(x):
    if x <= 25:
        return 1
    elif 25 < x <= 40:
        return 2
    elif 40 < x <= 60:
        return 3
    elif x > 60:
        return 4


training_dataset['age'] = training_dataset['age'].apply(categorize_age)
training_dataset['capital-gain'] = training_dataset['capital-gain'].apply(categorize_gain)
training_dataset['capital-loss'] = training_dataset['capital-loss'].apply(categorize_loss)
training_dataset['hours-per-week'] = training_dataset['hours-per-week'].apply(categorize_hours)

# Testing data
print(training_dataset)

# Model selection
X = training_dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].values
y = training_dataset.iloc[:, 8].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scaling the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Predict using SVC
clf = SVC(gamma=0.001, C=100.)

clf.fit(X=X_train, y=y_train)
myPred = clf.predict(X_test)

cmsvc = confusion_matrix(y_test, myPred)
print("Accuracy SVC:" + str(accuracy_score(y_test, myPred)) + str(cmsvc))

# Predict using RandomForest
classifier = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=50)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cmrf = confusion_matrix(y_test, y_pred)
print("Accuracy Random Forest:" + str(accuracy_score(y_test, y_pred)) + str(cmrf))

# Predict using NaiveBayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred3 = gnb.predict(X_test)

cmgnb = confusion_matrix(y_test, y_pred3)
print("Accuracy Gaussian:" + str(accuracy_score(y_test, y_pred3)) + str(cmgnb))

# Preparing test data
test_dataset = pd.read_csv("data/test.csv")
test_ids = test_dataset['Id']
del test_dataset['Id']
del test_dataset['fnlwgt']
del test_dataset['education']
del test_dataset['native-country']
del test_dataset['race']
del test_dataset['relationship']
del test_dataset['marital-status']

d1 = le.transform(test_dataset.iloc[:, [1, 3, 4]])
df2 = pd.DataFrame(d1)

test_dataset['workclass'] = df2[0]
test_dataset['occupation'] = df2[1]
test_dataset['sex'] = df2[2]

# Scaling data
X2_test = sc.transform(test_dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].values)

# Predict test using RandomForest using model trained above
y_final = classifier.predict(X2_test)

final = pd.DataFrame(test_ids)
final['Predicted'] = y_final

final.to_csv("data/results_forest.csv", index=False)

# Predict test using SVC using model trained above
y_final = clf.predict(X2_test)

final = pd.DataFrame(test_ids)
final['Predicted'] = y_final

final.to_csv("data/results_svc.csv", index=False)

# Predict test using NaiveBayes using model trained above
y_final = gnb.predict(X2_test)

final = pd.DataFrame(test_ids)
final['Predicted'] = y_final

final.to_csv("data/results_bayes.csv", index=False)