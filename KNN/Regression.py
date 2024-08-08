import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

# 농어(length, weight) 데이터
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
                         21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
                         23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
                         27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
                         39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
                         44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
                         115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
                         150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
                         218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
                         556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
                         850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
                         1000.0])

# 데이터 시각화
plt.scatter(perch_length, perch_weight)
plt.xlabel('Length (cm)')
plt.ylabel('Weight (g)')
plt.show()

# 훈련 세트와 테스트 세트로 데이터 분리
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
# 입력 데이터 형태 변경
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

# KNN 회귀 모델 생성 및 학습
knr = KNeighborsRegressor()
knr.fit(train_input, train_target)
score = knr.score(test_input, test_target)
print(f"R^2 score: {score}")

# 예측 및 MAE 계산
test_prediction = knr.predict(test_input)
mae = mean_absolute_error(test_target, test_prediction)
print(f"Mean Absolute Error: {mae}")

# 과대 적합, 과소 적합 확인
print(f"Training R^2 score: {knr.score(train_input, train_target)}")
print(f"Test R^2 score: {knr.score(test_input, test_target)}")

# 이웃 수를 3으로 설정하여 과소 적합 해결
knr.n_neighbors = 3
knr.fit(train_input, train_target)
print(f"Training R^2 score (k=3): {knr.score(train_input, train_target)}")
print(f"Test R^2 score (k=3): {knr.score(test_input, test_target)}")

# 회귀의 한계 확인 (입력값 범위 벗어남)
print(f"Predicted weight for length 100 cm: {knr.predict([[100]])}")

# 선형 회귀 모델 생성 및 학습
lr = LinearRegression()
lr.fit(train_input, train_target)
print(f"Predicted weight for length 50 cm: {lr.predict([[50]])}")
print(f"Linear Regression Coefficients: {lr.coef_}, Intercept: {lr.intercept_}")

# 선형 회귀 결과 시각화
plt.scatter(train_input, train_target)
plt.plot([15, 50], [15 * lr.coef_ + lr.intercept_, 50 * lr.coef_ + lr.intercept_], color='red')
plt.scatter(50, lr.predict([[50]]), marker='^')
plt.xlabel('Length (cm)')
plt.ylabel('Weight (g)')
plt.show()

# 선형 회귀 모델 평가
print(f"Training R^2 score (Linear Regression): {lr.score(train_input, train_target)}")
print(f"Test R^2 score (Linear Regression): {lr.score(test_input, test_target)}")

# 2차 다항 회귀를 위한 데이터 변환
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))

# 2차 다항 회귀 모델 생성 및 학습
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(f"Predicted weight for length 50 cm (Polynomial): {lr.predict([[50 ** 2, 50]])}")
print(f"Polynomial Regression Coefficients: {lr.coef_}, Intercept: {lr.intercept_}")

# 2차 다항 회귀 결과 시각화
point = np.arange(15, 50)
plt.scatter(train_input, train_target)
plt.plot(point, 1.01 * point ** 2 - 21.6 * point + lr.intercept_, color='red')
plt.scatter([50], lr.predict([[50 ** 2, 50]]), marker='^')
plt.xlabel('Length (cm)')
plt.ylabel('Weight (g)')
plt.show()

# 2차 다항 회귀 모델 평가
print(f"Training R^2 score (Polynomial Regression): {lr.score(train_poly, train_target)}")
print(f"Test R^2 score (Polynomial Regression): {lr.score(test_poly, test_target)}")
