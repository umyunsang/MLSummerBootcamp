import pandas as pd
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

wine = pd.read_csv('https://raw.githubusercontent.com/umyunsang/MLSummerBootcamp/master/wine.csv')
print(wine.head())

wine_input = wine[['alcohol', 'sugar', 'pH']].to_numpy()
print(wine_input[:5])

wine_target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(wine_input, wine_target, random_state=42)

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

sc = SGDClassifier(loss='log_loss', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

print(sc.predict(test_scaled[:5]))

lr = LogisticRegression()
lr.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))



