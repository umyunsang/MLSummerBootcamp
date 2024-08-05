import matplotlib.pyplot as plt
import numpy as np

# x 축 데이터 생성: 0부터 2π까지 200개의 점
x = np.linspace(0, 2 * np.pi, 200)

# y 축 데이터 생성: x의 사인 값
y = np.sin(x)

# 그림의 크기 설정
# figsize: 그림의 크기를 설정합니다. (가로, 세로) 단위는 인치입니다.
plt.figure(figsize=(4.2, 3.6))

# x와 y 데이터를 이용해 선 그래프를 그립니다.
plt.plot(x, y)

# 그리드 설정
# color: 그리드 선의 색상 설정
# linestyle: 그리드 선의 스타일을 점선으로 설정
# linewidth: 그리드 선의 두께를 설정
plt.grid(color='r', linestyle='dotted', linewidth=2)

# 그래프를 화면에 출력
plt.show()
