import matplotlib.pyplot as plt
import numpy as np

# 데이터와 관련된 정보
data = [5, 4, 6, 11]  # 각 조각의 크기
clist = ['cyan', 'gray', 'orange', 'red']  # 각 조각의 색상
explode = [.06, .07, .08, .09]  # 각 조각의 폭 (조각을 띄우는 정도)

# 파이 차트 생성
# autopct: 각 조각의 비율을 퍼센트로 표시
# colors: 각 조각의 색상
# labels: 각 조각의 레이블
# explode: 각 조각의 폭 (조각을 띄우는 정도)
plt.pie(data, autopct='%.2f', colors=clist, labels=clist, explode=explode)

# 그래프 출력
plt.show()
