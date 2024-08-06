import pandas as pd
import numpy as np

# 데이터 프레임을 생성하기 위한 Series 객체 생성
month_se = pd.Series(['1월', '2월', '3월', '4월'])  # 월을 나타내는 Series
income_se = pd.Series([9500, 6200, 6050, 7000])   # 각 월의 수익을 나타내는 Series
expenses_se = pd.Series([5040, 2350, 2300, 4800]) # 각 월의 지출을 나타내는 Series

# 데이터 프레임 생성
df = pd.DataFrame({
    '월': month_se,       # 월 열
    '수익': income_se,    # 수익 열
    '지출': expenses_se   # 지출 열
})

# 데이터 프레임 출력
print(df)

# 최대 수익이 발생한 월을 찾기
m_idx = np.argmax(income_se)  # 수익이 최대인 인덱스를 찾기

# 최대 수익이 발생한 월과 수익 출력
print('최대 수익이 발생한 월:', month_se[m_idx])  # 최대 수익이 발생한 월
print(f'월 최대 수익: {income_se.max()}, 월 평균 수익: {income_se.mean()}')  # 최대 수익과 평균 수익
