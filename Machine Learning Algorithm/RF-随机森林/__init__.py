# ------------------------------- #
# 集成学习 Ensemble Learning
# ------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)

plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.linear_model import LogisticRegression
log_clf = LogisticRegression()
log_clf.fit(X_train, y_train)
log_clf.score(X_test, y_test)

from sklearn.svm import SVC
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
svm_clf.score(X_test, y_test)

from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
dt_clf.score(X_test, y_test)

y_predict1 = log_clf.predict(X_test)
y_predict2 = svm_clf.predict(X_test)
y_predict3 = dt_clf.predict(X_test)

y_predict = np.array((y_predict1 + y_predict2 + y_predict3) >= 2, dtype='int')

y_predict[:10]

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predict)
# 集成学习效果更好

# 使用Voting Classifier
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(estimators=[
    ('log_clf', LogisticRegression()),
    ('svm_clf', SVC()),
    ('dt_clf', DecisionTreeClassifier(random_state=666))
], voting='hard')

voting_clf.fit(X_train, y_train)
voting_clf.score(X_test, y_test)


# ------------------------------- #
# SoftVoting Classifier
# ------------------------------- #
from sklearn.ensemble import VotingClassifier
voting_clf2 = VotingClassifier(estimators=[
    ('log_clf', LogisticRegression()),
    ('svm_clf', SVC(probability=True)),
    ('dt_clf', DecisionTreeClassifier(random_state=666))
], voting='soft')

voting_clf2.fit(X_train, y_train)
voting_clf2.score(X_test, y_test)


# ------------------------------- #
# Bagging和Pasting
# ------------------------------- #
# Bagging：放回取样（更常用）bootstrap
# Pasting：不放回取样
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

bagging_clf = BaggingClassifier(DecisionTreeClassifier(),
                                n_estimators=500, max_samples=100,
                                bootstrap=True)

bagging_clf.fit(X_train, y_train)
bagging_clf.score(X_test, y_test)

bagging_clf2 = BaggingClassifier(DecisionTreeClassifier(),
                                n_estimators=10000, max_samples=100,
                                bootstrap=True)

bagging_clf2.fit(X_train, y_train)
bagging_clf2.score(X_test, y_test)

# 尝试绘图
score = []
for i in range(1000,10000,500):
    bagging_clf = BaggingClassifier(DecisionTreeClassifier(),
                                n_estimators=i, max_samples=100,
                                bootstrap=True)
    bagging_clf.fit(X_train, y_train)
    score.append(bagging_clf.score(X_test, y_test))
    # plt.scatter(i, score)

x = np.linspace(1000,10000,18)
plt.plot(x, score)
plt.show()


# ------------------------------- #
# oob(out-of-bag)
# ------------------------------- #
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

bagging_clf = BaggingClassifier(DecisionTreeClassifier(),
                                n_estimators=500, max_samples=100,
                                bootstrap=True,oob_score=True)

bagging_clf.fit(X_train, y_train)

bagging_clf.oob_score_

# bootstrap_features
random_subspace_clf = BaggingClassifier(DecisionTreeClassifier(),
                                n_estimators=500, max_samples=500,
                                bootstrap=True,oob_score=True,
                                n_jobs=-1,
                                max_features=1, bootstrap_features=True)

random_subspace_clf.fit(X, y)
random_subspace_clf.oob_score_

random_pacthes_clf = BaggingClassifier(DecisionTreeClassifier(),
                                n_estimators=500, max_samples=100,
                                bootstrap=True,oob_score=True,
                                n_jobs=-1,
                                max_features=1, bootstrap_features=True)

random_pacthes_clf.fit(X, y)
random_pacthes_clf.oob_score_


# ------------------------------- #
# 随机森林和Extra-Tree
# ------------------------------- #
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, random_state=666, oob_score=True, n_jobs=-1)
rf_clf.fit(X, y)
rf_clf.oob_score_

# extra-tree: 使用随机的特征和随机的阈值
# 提供额外的随机性，抑制过拟合，更快的训练速度，但增大了bias
from sklearn.ensemble import ExtraTreesClassifier
et_clf = ExtraTreesClassifier(n_estimators=500, bootstrap=True, oob_score=True, random_state=666)
et_clf.fit(X, y)
et_clf.oob_score_


# ------------------------------- #
# Ada Boosting 和 Gradient Boosting
# ------------------------------- #
# Ada Boosting: 集成多个模型，每个模型都在尝试增强（Boosting）整体的效果
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=500)
ada_clf.fit(X_train, y_train)

ada_clf.score(X_test, y_test)

# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

gb_clf = GradientBoostingClassifier(max_depth=2, n_estimators=500)
gb_clf.fit(X_train, y_train)
gb_clf.score(X_test, y_test)

# Boosting 解决回归问题
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor


# ------------------------------- #
# Stacking
# ------------------------------- #
