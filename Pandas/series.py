import pandas as pd
import numpy as np

# pandas의 Series 객체를 생성하여 데이터를 저장하고 출력합니다.

# 단순한 Series 생성
se = pd.Series([1, 2, np.nan, 4])
print(se)  # Series 전체 출력
print(se[0])  # 인덱스 0의 값 출력
print(se[2])  # 인덱스 2의 값 출력
print(se.isna())  # 각 요소가 NaN인지 여부를 나타내는 Boolean Series 출력

# 인덱스를 지정하여 Series 생성
data = [1, 2, np.nan, 4]
indexed_se = pd.Series(data, index=['a', 'b', 'c', 'd'])
print(indexed_se)  # 인덱스가 지정된 Series 전체 출력
print(indexed_se['a'])  # 인덱스 'a'에 해당하는 값 출력
print(indexed_se['c'])  # 인덱스 'c'에 해당하는 값 출력

# 예제 문제: 학생들의 점수 데이터를 담은 Series 생성
scores = [78, 94, 56, 74, 67]
scores_se = pd.Series(scores, index=['김주연', '박효원', '정재현', '임승우', '황상필'])
print(scores_se)  # 학생들의 점수 데이터 출력
print(scores_se.mean())  # 점수의 평균 출력
print(scores_se.sum())  # 점수의 총합 출력
