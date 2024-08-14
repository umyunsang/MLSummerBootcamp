import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 와인 데이터셋을 CSV 파일에서 불러옵니다.
wine = pd.read_csv('https://raw.githubusercontent.com/umyunsang/MLSummerBootcamp/master/wine.csv')

# 데이터셋의 처음 5개 행을 출력하여 데이터 구조를 확인합니다.
print(wine.head())

# 입력 데이터로 사용할 'alcohol', 'sugar', 'pH' 열을 추출하여 numpy 배열로 변환합니다.
wine_input = wine[['alcohol', 'sugar', 'pH']].to_numpy()

# 입력 데이터의 처음 5개 행을 출력하여 확인합니다.
print(wine_input[:5])

# 목표 변수로 사용할 'class' 열을 추출하여 numpy 배열로 변환합니다.
wine_target = wine['class'].to_numpy()

# 데이터를 학습용(train)과 테스트용(test)으로 분할합니다.
train_input, test_input, train_target, test_target = train_test_split(wine_input, wine_target, random_state=42)

# DecisionTreeClassifier를 초기화하고 학습 데이터에 대해 학습시킵니다.
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_input, train_target)

# 학습 데이터와 테스트 데이터에 대한 모델의 정확도를 출력합니다.
print(dt.score(train_input, train_target))  # 학습 데이터 정확도
print(dt.score(test_input, test_target))    # 테스트 데이터 정확도

# 학습된 결정 트리(Decision Tree)를 시각화합니다.
plt.figure(figsize=(10, 7))
plot_tree(dt)
plt.show()

# 결정 트리의 최대 깊이를 1로 제한하고, 시각화를 개선하여 다시 그립니다.
plt.figure(figsize=(10, 7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

# **가지 치기(pruning)**를 사용하여 트리의 최대 깊이를 3으로 제한하고, 다시 모델을 학습시킵니다.
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)

# 가지 치기 후 학습 데이터와 테스트 데이터에 대한 모델의 정확도를 출력합니다.
print(dt.score(train_input, train_target))  # 학습 데이터 정확도
print(dt.score(test_input, test_target))    # 테스트 데이터 정확도

# 가지 치기된 트리를 시각화합니다.
plt.figure(figsize=(10, 7))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
