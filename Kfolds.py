import csv
import pandas as pd
from sklearn import svm
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

# load data
#training data
from sklearn.neighbors import KNeighborsClassifier

file = open('train_samples.txt', encoding="utf8")
data = [line.split('\t') for line in file.readlines()]

df_train_samples = pd.DataFrame()
df_train_samples['id'] = [row[0] for row in data]
df_train_samples['text'] = [row[1] for row in data]
df_train_samples.dropna(inplace = True)
df_train_labels = pd.read_csv('train_labels.txt', sep='\t', names=["id", "label"])

#internal validation data
file = open('validation_samples.txt', encoding="utf8")
data = [line.split('\t') for line in file.readlines()]

df_validation_samples = pd.DataFrame()
df_validation_samples['id'] = [row[0] for row in data]
df_validation_samples['text'] = [row[1] for row in data]
df_validation_samples.dropna(inplace=True)
df_validation_labels = pd.read_csv('validation_labels.txt', sep='\t', names=["id","label"])

print(df_validation_samples.shape)
print(df_validation_labels.shape)

#test data
file = open('test_samples.txt', encoding="utf8")
data = [line.split('\t') for line in file.readlines()]

df_test_samples = pd.DataFrame()
df_test_samples['id'] = [row[0] for row in data]
df_test_samples['text'] = [row[1] for row in data]
df_test_samples.dropna(inplace=True)

frames = [df_train_samples, df_validation_samples]
input = pd.concat(frames, sort=False)
frames1 = [df_train_labels, df_validation_labels]
data1 = pd.concat(frames1, sort=False)

#scaling/vectorize

vectorizer = TfidfVectorizer(min_df=0.001, max_df=0.6, ngram_range=(1, 7))
vectorizer.fit(input['text'])
input = vectorizer.transform(input['text'])
vectorizer.fit(df_train_samples['text'])
input1 = vectorizer.transform(df_train_samples['text'])
test = vectorizer.transform(df_test_samples['text'])
test1 = vectorizer.transform(df_validation_samples['text'])

df_validation_labels['label'] =(df_validation_labels['label']).astype(int)
data1['label'] = (data1['label']).astype(int)



digits = np.column_stack((df_train_samples['text'], df_train_labels['label']))
kf = KFold(n_splits=3)

# liste de scoruri
scores_svm1 = []
scores_svm2 = []
scores_svm3 = []
scores_svm4 = []
scores_comp1 = []
scores_comp2 = []
scores_comp3 = []
scores_comp4 = []
scores_lrg1 = []
scores_lrg2 = []
scores_vt = []

#o functie care intoarce scorul modelului model cu datele de antrenare train_text, train_label
#si datele de test test_text, test_label
def get_score(model, train_text, train_label, test_text, test_label):
    model.fit(train_text, train_label)
    pred = model.predict(test_text)
    return(f1_score(pred, test_label))


#pentru fiecare fold am testat diferiti clasificatori
for train_i, test_i in kf.split(input1):
    train_text = input1[train_i]
    train_label = df_train_labels['label'][train_i]
    test_text = input1[test_i]
    test_label = df_train_labels['label'][test_i]

    scores_lrg1.append(get_score(LogisticRegression(C=1,penalty='l2'), train_text, train_label, test_text,test_label))
    scores_lrg2.append(get_score(LogisticRegression(random_state=0, solver='saga', penalty='elasticnet', l1_ratio=1), train_text, train_label, test_text, test_label))
    scores_svm1.append(get_score(svm.SVC(kernel='linear', C=0.1, probability=True), train_text, train_label, test_text, test_label))
    scores_svm2.append(get_score(svm.SVC(kernel='linear', C=1,probability=True), train_text, train_label, test_text, test_label))
    scores_vt.append(get_score(VotingClassifier(estimators=[('cmp', ComplementNB(alpha=0.1)), ('rnf', LogisticRegression(random_state=0, solver='saga', penalty='elasticnet', l1_ratio=1)), ('svm', svm.SVC(kernel='linear', C=0.1, probability=True))], voting='hard'), train_text, train_label, test_text, test_label))
    scores_comp1.append(get_score(ComplementNB(alpha=1), train_text, train_label, test_text, test_label))
    scores_comp2.append(get_score(ComplementNB(alpha=0.1), train_text, train_label, test_text, test_label))
    scores_comp3.append(get_score(ComplementNB(alpha=0.4), train_text, train_label, test_text, test_label))
    # scores_comp3.append(get_score(ComplementNB(alpha=0), train_text, train_label, test_text, test_label))
    scores_comp4.append(get_score(MultinomialNB(alpha=0.1), train_text, train_label, test_text, test_label))

alpha = [0, 0.1, 0.4, 1]
fold = [1, 2, 3]
#am preluat valorile pentru parametri diferiti pentru fiecare fold
#voi afisa acuratetea lor pe un grafic
fold1 = []
fold1.append(scores_comp1[0])
fold1.append(scores_comp2[0])
fold1.append(scores_comp3[0])
fold1.append(scores_comp4[0])

fold2 = []
fold2.append(scores_comp1[1])
fold2.append(scores_comp2[1])
fold2.append(scores_comp3[1])
fold2.append(scores_comp4[1])

fold3 = []
fold3.append(scores_comp1[2])
fold3.append(scores_comp2[2])
fold3.append(scores_comp3[2])
fold3.append(scores_comp4[2])

#Ploteaza punctele
plt.plot(alpha, fold1)
plt.plot(alpha, fold2)
plt.plot(alpha, fold3)
plt.gca().legend('123')
# Adauga etichete pentru fiecare axa
plt.xlabel('alpha')
plt.ylabel('f1_score')
# Afiseaza figura
plt.show()

#ComplementNB vs MultinomialNB vs SVM vs LogisticRegression
plt.plot(fold, scores_comp2)
plt.plot(fold, scores_svm1)
plt.plot(fold, scores_lrg2)
plt.gca().legend('CSL')
plt.xlabel('fold')
plt.ylabel('f1_score')
plt.show()

grid = {"C":np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, grid, cv=10)
logreg_cv.fit(input1, df_train_labels['label'])
print("tuned hpyerparameters: (best parameters) ", logreg_cv.best_params_)
print("accuracy: ", logreg_cv.best_score_)

#grafic pentru logistic regression
plt.plot(fold, scores_lrg1)
plt.plot(fold, scores_lrg2)
plt.gca().legend('12')
plt.xlabel('fold')
plt.ylabel('f1_score')
plt.show()

#Voting classifier
modelA = ComplementNB(alpha=0.1, class_prior=None, fit_prior=True, norm=False)
modelB = svm.SVC(kernel='linear', C=0.1)
modelC = LogisticRegression(random_state=0, solver ='saga', penalty='elasticnet', l1_ratio=1)

model2 = VotingClassifier(estimators=[('cmp', modelA), ('svm', modelB), ('lrg', modelC)], voting='hard')
model2.fit(input1, df_train_labels['label'])
pred2 = model2.predict(test1)
confusion_matrix = pd.crosstab(pred2, df_validation_labels['label'], rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
sn.heatmap(confusion_matrix, annot=True)
plt.show()

#In final dupa alegerea modelului cel mai eficient am concatenat datele de taining si de validare, am antrenat modelul pe ele dupa care l-am testat pe datele test (df_test_samples)
#ComplementNB
model1 = ComplementNB(alpha=0.1)
model1.fit(input, data['label'])
pred1 = model1.predict(test)

#scrierea rezultatelor
data = np.column_stack((df_test_samples['id'], pred1))
print(data.shape)

with open('submission1.csv', 'w', newline='') as csvfile:
    fieldnames = ['id', 'label']
    writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
    writer.writerow({'id': 'id', 'label':'label'})
    for row in data:
        writer.writerow({'id': row[0], 'label': row[1]})
