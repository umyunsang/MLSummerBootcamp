import matplotlib.pyplot as plt
import numpy as np

# 데이터 포인트 개수
N = 30

# 랜덤한 x 좌표와 y 좌표 생성
x = np.random.rand(N)  # 0과 1 사이의 랜덤 값 N개
y = np.random.rand(N)  # 0과 1 사이의 랜덤 값 N개

# 점의 색상을 위한 랜덤 값 생성
color = np.random.rand(N)  # 0과 1 사이의 랜덤 값 N개

# 점의 크기를 위한 랜덤 값 생성
# (30 * 랜덤 값)^2로 점의 크기를 결정
area = (30 * np.random.rand(N)) ** 2  # 0부터 30 사이의 크기

# 산점도 그래프 생성
# s: 점의 크기, c: 점의 색상, alpha: 투명도
plt.scatter(x, y, s=area, c=color, alpha=0.5)

# 그래프 출력
plt.show()
