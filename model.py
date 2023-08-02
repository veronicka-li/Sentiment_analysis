from preprocess import data_full

# # Деление на тренировочную и тестовую выборку V2

X = data_full['vec'].to_list()
y = data_full['sentiment'].to_list()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y)

# # Обучение
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

classifier.fit(X_train,y_train)


from sklearn import metrics
predicted = classifier.predict(X_test)
accuracy_score = metrics.accuracy_score(predicted, y_test)
f1_score = metrics.f1_score(predicted, y_test)


print(str('{:.1%}'.format(accuracy_score)))
print(str('{:.1%}'.format(f1_score)))



