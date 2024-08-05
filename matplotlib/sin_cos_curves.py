import matplotlib.pyplot as plt
import numpy as np

# 0부터 2*pi까지 100개의 점 생성
x = np.linspace(0, np.pi * 2, 100)

# 어두운 배경 스타일 적용
plt.style.use('dark_background')

# 그래프 제목 설정
plt.title('사인과 코사인 곡선')

# 빨간 실선으로 사인 곡선 그리기, 파란 점선으로 코사인 곡선 그리기
plt.plot(x, np.sin(x), 'r-', label='사인 곡선')
plt.plot(x, np.cos(x), 'b:', label='코사인 곡선')

# x축과 y축 레이블 설정
plt.xlabel('x 값')
plt.ylabel('y 값')

# 범례 표시
plt.legend()

# 그래프 출력
plt.show()
