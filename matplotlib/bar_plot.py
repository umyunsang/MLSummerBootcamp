import matplotlib.pyplot as plt
import numpy as np

# x 축의 위치를 지정하는 배열
x = np.arange(3)  # [0, 1, 2]

# x 축에 표시할 연도
years = ['2010', '2011', '2012']

# 국내 데이터
domestic = [6801, 7695, 8010]  # 각 연도의 국내 데이터

# 해외 데이터
foreign = [777, 1046, 1681]  # 각 연도의 해외 데이터

# 국내 데이터에 대한 막대 그래프 생성
plt.bar(x, domestic, width=0.25, label='Domestic')

# 해외 데이터에 대한 막대 그래프 생성
# x 위치를 0.3만큼 이동하여 막대가 겹치지 않도록 함
plt.bar(x + 0.3, foreign, width=0.25, label='Foreign')

# x 축의 눈금과 레이블 설정
plt.xticks(x + 0.15, years)  # x 위치를 막대의 중앙에 맞추기 위해 x + 0.15로 조정

# 범례 추가
plt.legend()

# 그래프 출력
plt.show()
