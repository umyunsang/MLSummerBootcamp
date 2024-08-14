import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from matplotlib import font_manager, rc

# 한글 폰트 설정 (Windows)
font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows에서는 맑은 고딕 폰트 사용
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

# 데이터셋 불러오기
path = 'https://raw.githubusercontent.com/umyunsang/MLSummerBootcamp/master/perch_full.csv'
df = pd.read_csv(path)
perch_full = df.to_numpy()

# 타겟 변수: 농어의 무게
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
                         115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
                         150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
                         218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
                         556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
                         850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
                         1000.0])

# 데이터 분할: 훈련 데이터셋과 테스트 데이터셋
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)

# 다항 특성 추가
poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)

# 다항 회귀 모델 훈련 및 성능 평가
lr = LinearRegression()
lr.fit(train_poly, train_target)
train_score = lr.score(train_poly, train_target)
test_score = lr.score(test_poly, test_target)
print(f"다항 회귀 (degree=5) 훈련 세트 R^2 점수: {train_score:.2f}")
print(f"다항 회귀 (degree=5) 테스트 세트 R^2 점수: {test_score:.2f}")

# 데이터 표준화
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

# 릿지 회귀: 적절한 규제 강도(alpha) 찾기
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    ridge = Ridge(alpha=alpha)
    ridge.fit(train_scaled, train_target)
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))

# 규제 강도(alpha)에 따른 성능 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(np.log10(alpha_list), train_score, label='훈련 세트')
plt.plot(np.log10(alpha_list), test_score, label='테스트 세트')
plt.xlabel('log(alpha)')
plt.ylabel('R^2')
plt.title('릿지 회귀: alpha에 따른 성능 변화')
plt.legend()
plt.show()

# 최적의 alpha 값으로 릿지 회귀 모델 훈련 및 평가
best_alpha = 0.1
ridge = Ridge(alpha=best_alpha)
ridge.fit(train_scaled, train_target)
train_score = ridge.score(train_scaled, train_target)
test_score = ridge.score(test_scaled, test_target)
print(f"릿지 회귀 (alpha={best_alpha}) 훈련 세트 R^2 점수: {train_score:.2f}")
print(f"릿지 회귀 (alpha={best_alpha}) 테스트 세트 R^2 점수: {test_score:.2f}")

# 라쏘 회귀: 적절한 규제 강도(alpha) 찾기
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    lasso = Lasso(alpha=alpha)
    lasso.fit(train_scaled, train_target)
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))

# 규제 강도(alpha)에 따른 성능 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(np.log10(alpha_list), train_score, label='훈련 세트')
plt.plot(np.log10(alpha_list), test_score, label='테스트 세트')
plt.xlabel('log(alpha)')
plt.ylabel('R^2')
plt.title('라쏘 회귀: alpha에 따른 성능 변화')
plt.legend()
plt.show()

# 최적의 alpha 값으로 라쏘 회귀 모델 훈련 및 평가
best_alpha = 10
lasso = Lasso(alpha=best_alpha)
lasso.fit(train_scaled, train_target)
train_score = lasso.score(train_scaled, train_target)
test_score = lasso.score(test_scaled, test_target)
print(f"라쏘 회귀 (alpha={best_alpha}) 훈련 세트 R^2 점수: {train_score:.2f}")
print(f"라쏘 회귀 (alpha={best_alpha}) 테스트 세트 R^2 점수: {test_score:.2f}")

# 계수가 0인 특성의 개수 확인
zero_coef_count = np.sum(lasso.coef_ == 0)
print(f"라쏘 회귀 (alpha={best_alpha})에서 계수가 0인 특성의 개수: {zero_coef_count}")
