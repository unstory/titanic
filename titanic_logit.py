#coding=utf-8
import pandas as pd
import numpy as np
import sys

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
# Embarked,drop "s"
train["Embarked"] = train.Embarked.fillna("S")
embarked_dummies_train = pd.get_dummies(train.Embarked)
train.drop("Embarked",axis=1, inplace=True)
train = train.join(embarked_dummies_train)
train.drop("S", axis=1, inplace=True)

test["Embarked"] = test.Embarked.fillna("S")
embarked_dummies_test = pd.get_dummies(test.Embarked)
test.drop("Embarked",axis=1, inplace=True)
test = test.join(embarked_dummies_test)
test.drop("S", axis=1, inplace=True)

# Cabin, drop
train.drop("Cabin", axis=1, inplace=True)
test.drop("Cabin", axis=1, inplace=True)

# age, change to "child", "young", "old"
train = train[train.Age.notnull()]
avg_age_test = np.mean(test.Age)
std_age_test = np.std(test.Age)
nan_count = test.Age.isnull().sum()
rand_test = np.random.randint(avg_age_test - std_age_test, avg_age_test + std_age_test,size=nan_count)
test.loc[test.Age.isnull(), ["Age"]] = rand_test
def grade(age):
    if int(age) <= 18:
        return "child"
    elif int(age) < 60:
        return "young"
    else:
        return "old"
train["Person"] = train.Age.apply(grade)
train.drop("Age", axis=1, inplace=True)
person_dummies_train = pd.get_dummies(train.Person)
train = train.join(person_dummies_train)
train.drop("Person", axis=1, inplace=True)

test["Person"] = test.Age.apply(grade)
test.drop("Age", axis=1, inplace=True)
person_dummies_test = pd.get_dummies(test.Person)
test = test.join(person_dummies_test)
test.drop("Person", axis=1, inplace=True)
# sex
sex_dummies_train = pd.get_dummies(train.Sex)
train = train.join(sex_dummies_train)
train.drop("Sex", axis=1, inplace=True)
sex_dummies_test = pd.get_dummies(test.Sex)
test = test.join(sex_dummies_test)
test.drop("Sex", axis=1, inplace=True)

# SibSp, Parch
train["Family"] = train.SibSp + train.Parch
test["Family"] = test.SibSp + test.Parch
def countPerson(x):
    if x > 0:
        return 1
    else:
        return 0
train["Family"] = train.Family.apply(countPerson)
test["Family"] = test.Family.apply(countPerson)
train.drop(["SibSp", "Parch"], axis=1, inplace=True)
test.drop(["SibSp", "Parch"], axis=1, inplace=True)

pclass_dummies_train = pd.get_dummies(train.Pclass, prefix="pclass")
train = train.join(pclass_dummies_train)
train.drop("Pclass", axis=1, inplace=True)
pclass_dummies_test = pd.get_dummies(test.Pclass, prefix="pclass")
test = test.join(pclass_dummies_test)
test.drop("Pclass", axis=1, inplace=True)

train.drop(["Fare", "Ticket", "Name", "PassengerId"], axis=1, inplace=True)
test.drop(["Fare", "Ticket", "Name"], axis=1, inplace=True)
X_train = train.drop("Survived", axis=1)
Y_train = train.Survived

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# logreg = LogisticRegression()
# logreg.fit(X_train, Y_train)
# Y_pred = logreg.predict(test.drop("PassengerId", axis=1))
# print(logreg.score(X_train, Y_train))
random_forests = RandomForestClassifier(n_estimators=100)
random_forests.fit(X_train, Y_train)
Y_pred = random_forests.predict(test.drop("PassengerId", axis=1))
print(random_forests.score(X_train, Y_train))
submission = pd.DataFrame({"PassengerId":test["PassengerId"], "Survived":Y_pred})
submission.to_csv("result.csv", index=False)
print("Success")


