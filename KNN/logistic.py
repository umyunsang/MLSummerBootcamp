import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# CSV 파일에서 데이터셋을 불러옵니다.
path = 'https://raw.githubusercontent.com/umyunsang/MLSummerBootcamp/master/fish.csv'
fish = pd.read_csv(path)

# 데이터셋에 있는 고유한 어종(Species)을 출력하여 목표 변수를 확인합니다.
print(pd.unique(fish['Species']))

# 모델의 입력 변수(특징)를 선택합니다.
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()

# 입력 데이터의 첫 5개 샘플을 출력하여 데이터를 확인합니다.
print(fish_input[:5])

# 모델의 목표 변수(어종)를 선택합니다.
fish_target = fish['Species'].to_numpy()

# 데이터를 훈련 세트와 테스트 세트로 분리합니다.
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

# 입력 데이터를 표준화(정규화)하기 위한 StandardScaler를 생성합니다.
ss = StandardScaler()

# 훈련 데이터를 기준으로 표준화 기준을 학습(fit)합니다.
ss.fit(train_input)

# 학습된 표준화 기준을 이용하여 훈련 데이터와 테스트 데이터를 변환(transform)합니다.
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# K-최근접 이웃(K-Nearest Neighbors, KNN) 분류기를 생성하고, 이웃의 수를 3으로 설정합니다.
kn = KNeighborsClassifier(n_neighbors=3)

# 표준화된 훈련 데이터를 사용해 KNN 모델을 학습시킵니다.
kn.fit(train_scaled, train_target)

# 훈련 세트에서의 모델 정확도를 출력합니다.
print(kn.score(train_scaled, train_target))

# 테스트 세트에서의 모델 정확도를 출력합니다.
print(kn.score(test_scaled, test_target))

# 학습된 모델이 분류할 수 있는 클래스(어종)을 출력합니다.
print(kn.classes_)

# 테스트 데이터의 첫 5개 샘플에 대한 예측 결과를 출력합니다.
print(kn.predict(test_scaled[:5]))

# 테스트 데이터의 첫 5개 샘플에 대한 각 클래스의 예측 확률을 출력합니다.
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))

# 시그모이드 함수 그래프를 그리기 위한 z 값 생성 (-5에서 5까지 0.1 간격으로)
z = np.arange(-5, 5, 0.1)

# 시그모이드 함수 적용
phi = 1 / (1 + np.exp(-z))

# 시그모이드 함수 그래프를 그립니다.
plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()

# 도미(Bream)와 빙어(Smelt) 데이터만 추출
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

# 로지스틱 회귀 모델 생성 (이진 분류)
lr = LogisticRegression()

# 도미와 빙어 데이터를 사용하여 로지스틱 회귀 모델을 학습시킵니다.
lr.fit(train_bream_smelt, target_bream_smelt)

# 첫 5개 샘플에 대한 예측 결과를 출력합니다.
print(lr.predict(train_bream_smelt[:5]))

# 학습된 모델의 클래스(도미와 빙어)를 출력합니다.
print(lr.classes_)

# 로지스틱 회귀 모델의 계수와 절편을 출력합니다.
print(lr.coef_, lr.intercept_)

# 첫 5개 샘플에 대한 결정 함수 값을 출력합니다.
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)

# 결정 함수 값을 시그모이드 함수를 적용하여 확률로 변환합니다.
print(expit(decisions))

# 다중 클래스 분류를 위한 로지스틱 회귀 모델 생성
lr = LogisticRegression(C=20, max_iter=1000)

# 전체 데이터를 사용하여 로지스틱 회귀 모델을 학습시킵니다.
lr.fit(train_scaled, train_target)

# 훈련 세트에서의 모델 정확도를 출력합니다.
print(lr.score(train_scaled, train_target))

# 테스트 세트에서의 모델 정확도를 출력합니다.
print(lr.score(test_scaled, test_target))

# 테스트 데이터의 첫 5개 샘플에 대한 예측 결과를 출력합니다.
print(lr.predict(test_scaled[:5]))

# 학습된 모델의 클래스(어종)을 출력합니다.
print(lr.classes_)

# 첫 5개 샘플에 대한 각 클래스의 예측 확률을 출력합니다.
proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))

# 로지스틱 회귀 모델의 계수와 절편의 형태를 출력합니다.
print(lr.coef_.shape, lr.intercept_.shape)

# 첫 5개 샘플에 대한 결정 함수 값을 출력합니다.
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))

# 소프트맥스 함수를 적용하여 각 클래스에 대한 확률로 변환합니다.
proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))
