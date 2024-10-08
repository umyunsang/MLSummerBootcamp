
---
#### **1. 서브플롯 생성**

- **목표**: 2x2 그리드로 서브플롯을 생성합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 2x2 그리드로 서브플롯 생성
fig, ax = plt.subplots(2, 2)
```

- **설명**:
  - `fig`: 전체 그림 객체를 나타냅니다.
  - `ax`: 각 서브플롯의 축을 나타내는 배열입니다. `ax[0, 0]`, `ax[0, 1]`, `ax[1, 0]`, `ax[1, 1]`로 접근할 수 있습니다.

---

#### **2. 첫 번째 서브플롯: 산점도**

- **목표**: 정규 분포에서 생성된 데이터의 산점도를 그립니다.

```python
X = np.random.randn(100)  # 평균 0, 표준편차 1인 정규 분포에서 100개의 난수 생성
Y = np.random.randn(100)  # 평균 0, 표준편차 1인 정규 분포에서 100개의 난수 생성
ax[0, 0].scatter(X, Y)  # 산점도 그래프 생성
```

- **설명**:
  - `np.random.randn(100)`: 평균 0, 표준편차 1인 정규 분포에서 100개의 난수를 생성합니다.
  - `ax[0, 0].scatter(X, Y)`: 첫 번째 서브플롯에서 산점도를 그립니다.

---

#### **3. 두 번째 서브플롯: 막대 그래프**

- **목표**: 균등 분포에서 생성된 데이터의 막대 그래프를 그립니다.

```python
X = np.arange(10)  # 0부터 9까지의 정수 배열 생성
Y = np.random.uniform(1, 10, 10)  # 1부터 10 사이의 균등 분포에서 10개의 난수 생성
ax[0, 1].bar(X, Y)  # 막대그래프 생성
```

- **설명**:
  - `np.arange(10)`: 0부터 9까지의 정수 배열을 생성합니다.
  - `np.random.uniform(1, 10, 10)`: 1부터 10 사이의 균등 분포에서 10개의 난수를 생성합니다.
  - `ax[0, 1].bar(X, Y)`: 두 번째 서브플롯에서 막대 그래프를 그립니다.

---

#### **4. 세 번째 서브플롯: 선 그래프**

- **목표**: 코사인 함수를 사용한 선 그래프를 그립니다.

```python
X = np.linspace(0, 10, 100)  # 0부터 10까지 100개의 점 생성
Y = np.cos(X)  # X에 대한 코사인 함수 값 계산
ax[1, 0].plot(X, Y)  # 선 그래프 생성
```

- **설명**:
  - `np.linspace(0, 10, 100)`: 0부터 10까지 100개의 점을 생성합니다.
  - `np.cos(X)`: X에 대한 코사인 값을 계산합니다.
  - `ax[1, 0].plot(X, Y)`: 세 번째 서브플롯에서 선 그래프를 그립니다.

---

#### **5. 네 번째 서브플롯: 이미지**

- **목표**: 무작위로 생성된 데이터의 이미지를 표시합니다.

```python
Z = np.random.uniform(0, 1, (5, 5))  # 0부터 1 사이의 균등 분포에서 5x5 배열 생성
ax[1, 1].imshow(Z)  # 이미지를 서브플롯에 표시
```

- **설명**:
  - `np.random.uniform(0, 1, (5, 5))`: 0부터 1 사이의 균등 분포에서 5x5 배열을 생성합니다.
  - `ax[1, 1].imshow(Z)`: 네 번째 서브플롯에서 이미지를 표시합니다.

---

