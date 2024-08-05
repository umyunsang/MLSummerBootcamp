import matplotlib.pyplot as plt
import numpy as np

# 10x10 크기의 랜덤 데이터 배열 생성
data = np.random.random((10, 10))

# 데이터 배열을 이미지 형태로 표시
# cmap='Greys': 그레이스케일 색상 맵을 사용하여 데이터 값을 색상으로 변환
plt.imshow(data, cmap='Greys')

# 색상 막대를 추가하여 데이터 값과 색상 간의 매핑을 표시
plt.colorbar()

# 그래프 출력
plt.show()
