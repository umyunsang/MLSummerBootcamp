import matplotlib.pyplot as plt
import numpy as np

# 랜덤 시드 설정 (재현성을 위해)
np.random.seed(42)

# 데이터 생성
data1 = np.random.normal(100, 10, 200)  # 평균 100, 표준편차 10, 200개의 데이터
data2 = np.random.normal(100, 40, 200)  # 평균 100, 표준편차 40, 200개의 데이터
data3 = np.random.normal(80, 40, 200)   # 평균 80, 표준편차 40, 200개의 데이터
data4 = np.random.normal(80, 60, 200)   # 평균 80, 표준편차 60, 200개의 데이터

# 박스 플롯 생성
# 여러 데이터 세트를 비교하기 위해 리스트 형태로 전달
plt.boxplot([data1, data2, data3, data4])

# 그래프 출력
plt.show()
