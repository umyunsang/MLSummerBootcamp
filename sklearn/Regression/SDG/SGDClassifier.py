import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 제공된 URL에서 물고기 데이터셋을 불러옵니다.
path = 'https://raw.githubusercontent.com/umyunsang/MLSummerBootcamp/master/fish.csv'
fish = pd.read_csv(path)

# 입력 데이터(Weight, Length, Diagonal, Height, Width)를 numpy 배열로 변환합니다.
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()

# 목표 변수(Species)를 numpy 배열로 변환합니다.
fish_target = fish['Species'].to_numpy()

# 데이터를 학습용(train)과 테스트용(test)으로 나눕니다.
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

# 표준화(정규화)를 위한 스케일러를 초기화합니다.
ss = StandardScaler()

# 학습 데이터에 대해 스케일러를 학습시킵니다.
ss.fit(train_input)

# 학습 데이터와 테스트 데이터를 표준화합니다.
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# SGDClassifier를 로그 손실 함수(log_loss)를 사용하여 초기화하고 학습합니다.
sc = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
sc.fit(train_scaled, train_target)

# 학습 데이터와 테스트 데이터에 대한 모델의 정확도를 출력합니다.
print(sc.score(train_scaled, train_target))  # 학습 데이터 정확도
print(sc.score(test_scaled, test_target))    # 테스트 데이터 정확도

# 새로운 SGDClassifier 객체를 생성하여 부분 학습(점진적 학습)을 수행합니다.
sc = SGDClassifier(loss='log_loss', random_state=42)
train_score = []
test_score = []
classes = np.unique(train_target)

# 300번의 epoch 동안 부분 학습을 수행하며, 각 epoch에서의 정확도를 기록합니다.
for _ in range(0, 300):
    sc.partial_fit(train_scaled, train_target, classes=classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))

# 학습 과정에서의 학습 데이터와 테스트 데이터의 정확도 변화를 시각화합니다.
plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')   # x축 라벨: epoch
plt.ylabel('accuracy')  # y축 라벨: 정확도
plt.show()

# SGDClassifier를 최대 100번의 반복(iteration) 동안 학습시키고, tol=None을 설정하여 조기 종료를 비활성화합니다.
sc = SGDClassifier(loss='log_loss', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)

# 학습 데이터와 테스트 데이터에 대한 모델의 최종 정확도를 출력합니다.
print(sc.score(train_scaled, train_target))  # 학습 데이터 정확도
print(sc.score(test_scaled, test_target))    # 테스트 데이터 정확도

# 이번에는 힌지 손실 함수(hinge loss)를 사용하여 SGDClassifiers 학습합니다.
sc = SGDClassifier(loss='hinge', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)

# 학습 데이터와 테스트 데이터에 대한 모델의 최종 정확도를 출력합니다.
print(sc.score(train_scaled, train_target))  # 학습 데이터 정확도
print(sc.score(test_scaled, test_target))    # 테스트 데이터 정확도
