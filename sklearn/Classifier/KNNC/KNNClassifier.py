import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# 도미 35마리 length길이(cm)와 weight무게(g) 데이터
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 빙어 14마리 length길이(cm)와 weight무게(g) 데이터
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 도미와 빙어 데이터를 시각화
plt.scatter(bream_length, bream_weight, label='Bream')
plt.scatter(smelt_length, smelt_weight, label='Smelt')
plt.xlabel('Length (cm)')
plt.ylabel('Weight (g)')
plt.legend()
plt.show()

# 도미와 빙어의 길이와 무게 데이터 통합
fish_length = bream_length + smelt_length
fish_weight = bream_weight + smelt_weight

# 2차원 배열로 변환
fish_data = np.column_stack((fish_length, fish_weight))
# 도미(1)와 빙어(0) 레이블 생성
fish_target = np.concatenate((np.ones(35), np.zeros(14)))

# 데이터 확인
print(fish_data)
print(fish_target)

# sklearn 분류기 생성 및 학습
kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)
score = kn.score(fish_data, fish_target)
print(f"Training accuracy: {score}")

# 새로운 데이터 예측
print(f"Predicted label for [30, 600]: {kn.predict([[30, 600]])}")
print(f"Predicted label for [25, 150]: {kn.predict([[25, 150]])}")

# 데이터셋 분리 (훈련 세트와 테스트 세트)
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)

# 분리된 데이터를 시각화
plt.scatter(train_input[:, 0], train_input[:, 1], label='Train')
plt.scatter(test_input[:, 0], test_input[:, 1], label='Test')
plt.xlabel('Length (cm)')
plt.ylabel('Weight (g)')
plt.legend()
plt.show()

# 훈련 데이터로 sklearn 분류기 학습 및 테스트 데이터 평가
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
score = kn.score(test_input, test_target)
print(f"Test accuracy: {score}")
print(f"Predicted label for [25, 150]: {kn.predict([[25, 150]])}")

# 데이터 표준화 (Standardization)
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
train_scaled = (train_input - mean) / std
new = ([25, 150] - mean) / std

# 표준화된 데이터 시각화
plt.scatter(train_scaled[:, 0], train_scaled[:, 1], label='Train')
plt.scatter(new[0], new[1], marker='^', label='New')
plt.xlabel('Length (standardized)')
plt.ylabel('Weight (standardized)')
plt.legend()
plt.show()

# 표준화된 데이터로 sklearn 분류기 학습
kn = KNeighborsClassifier()
kn.fit(train_scaled, train_target)

# 테스트 데이터 표준화
test_scaled = (test_input - mean) / std
score = kn.score(test_scaled, test_target)
print(f"Test accuracy (scaled): {score}")

# 새로운 데이터 예측
print(f"Predicted label for scaled [25, 150]: {kn.predict([new])}")

# 이웃 데이터 확인 및 시각화
distances, indexes = kn.kneighbors([new])

plt.scatter(train_scaled[:, 0], train_scaled[:, 1], label='Train')
plt.scatter(new[0], new[1], marker='^', label='New')
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker='D', label='Neighbors')
plt.xlabel('Length (standardized)')
plt.ylabel('Weight (standardized)')
plt.legend()
plt.show()

