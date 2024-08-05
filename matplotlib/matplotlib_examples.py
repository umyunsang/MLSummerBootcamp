import matplotlib.pyplot as plt
import numpy as np

# 기본 라인 그래프 그리기 (기본 x 값 [0, 1, 2, 3])
plt.plot([1, 2, 3, 4])
plt.ylabel('y축')  # y축 레이블 설정
plt.xlabel('x축')  # x축 레이블 설정
plt.show()  # 그래프 출력

# 넘파이를 사용하여 x^2 그래프 그리기
x = np.arange(10)
plt.plot(x ** 2)
plt.show()  # 그래프 출력

# x^2를 그리고 축 범위를 [0, 100, 0, 100]으로 설정하여 그리기
x = np.arange(10)
plt.plot(x ** 2)
plt.axis([0, 100, 0, 100])
plt.show()  # 그래프 출력

# 여러 함수들을 그래프로 그리기: y1 = 2*x, y2 = (1/3)*x^2 + 5, y3 = -x^2 + 5
x = np.arange(-20, 20)
y1 = 2 * x
y2 = (1 / 3) * x ** 2 + 5
y3 = -x ** 2 + 5
plt.plot(x, y1, 'g--', y2, 'r^-', x, y3, 'b*:')  # 각 그래프 스타일 설정
plt.axis([-30, 30, -30, 30])  # 축 범위 설정
plt.show()  # 그래프 출력

# sin(x)와 cos(x)를 같은 그래프에 그리기
x = np.linspace(0, np.pi * 2, 100)
plt.plot(x, np.sin(x), 'r-')  # sin(x)을 빨간 실선으로 그리기
plt.plot(x, np.cos(x), 'b:')  # cos(x)를 파란 점선으로 그리기
plt.show()  # 그래프 출력

# 그래프를 'view.png' 파일로 저장하기
x = np.linspace(0, np.pi * 2, 100)
fig = plt.figure()
plt.plot(x, np.sin(x), 'r-')  # sin(x)을 빨간 실선으로 그리기
plt.plot(x, np.cos(x), 'b:')  # cos(x)를 파란 점선으로 그리기
fig.savefig('view.png')  # 'view.png'로 그림 저장
