import matplotlib.pyplot as plt
import numpy as np

# 2x2 그리드로 서브플롯 생성
fig, ax = plt.subplots(2, 2)

# 첫 번째 서브플롯: 산점도
X = np.random.randn(100)  # 평균 0, 표준편차 1인 정규 분포에서 100개의 난수 생성
Y = np.random.randn(100)  # 평균 0, 표준편차 1인 정규 분포에서 100개의 난수 생성
ax[0, 0].scatter(X, Y)  # 산점도 그래프 생성

# 두 번째 서브플롯: 막대그래프
X = np.arange(10)  # 0부터 9까지의 정수 배열 생성
Y = np.random.uniform(1, 10, 10)  # 1부터 10 사이의 균등 분포에서 10개의 난수 생성
ax[0, 1].bar(X, Y)  # 막대그래프 생성

# 세 번째 서브플롯: 선 그래프
X = np.linspace(0, 10, 100)  # 0부터 10까지 100개의 점 생성
Y = np.cos(X)  # X에 대한 코사인 함수 값 계산
ax[1, 0].plot(X, Y)  # 선 그래프 생성

# 네 번째 서브플롯: 이미지
Z = np.random.uniform(0, 1, (5, 5))  # 0부터 1 사이의 균등 분포에서 5x5 배열 생성
ax[1, 1].imshow(Z)  # 이미지를 서브플롯에 표시

# 모든 서브플롯을 화면에 출력
plt.show()
