import matplotlib.pyplot as plt

# 2x3 그리드로 서브플롯 생성
fig, ax = plt.subplots(2, 3)

# 각 서브플롯에 텍스트 표시
for i in range(2):
    for j in range(3):
        # (0.3, 0.5) 위치에 (i, j) 튜플을 텍스트로 표시
        # fontsize=11로 폰트 크기를 설정
        ax[i, j].text(0.3, 0.5, str((i, j)), fontsize=11)

# 모든 서브플롯을 화면에 출력
plt.show()
