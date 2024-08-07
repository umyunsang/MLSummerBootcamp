import pandas as pd
import numpy as np

# DataFrame 생성
df = pd.DataFrame({
    '상품': ['시계', '반지', '반지', '목걸이', '팔찌'],
    '재질': ['금', '은', '백금', '금', '은'],
    '가격': [500000, 20000, 350000, 300000, 60000]
})
print(df)

# pivot을 사용하여 '상품'을 행, '재질'을 열, '가격'을 값으로 하는 새로운 DataFrame 생성
new_df = df.pivot(index='상품', columns='재질', values='가격')
# 결측치를 0으로 채움
new_df = new_df.fillna(value=0)
print(new_df)

# 첫 번째 DataFrame 생성
df_1 = pd.DataFrame({
    'A': ['a10', 'a11', 'a12'],
    'B': ['b10', 'b11', 'b12'],
    'C': ['c10', 'c11', 'c12']
}, index=['가', '나', '다'])

# 두 번째 DataFrame 생성
df_2 = pd.DataFrame({
    'B': ['b23', 'b24', 'b25'],
    'C': ['c23', 'c24', 'c25'],
    'D': ['d23', 'd24', 'd25']
}, index=['다', '라', '마'])

# 두 DataFrame을 세로로 결합
df_3 = pd.concat([df_1, df_2])
print(df_3)

# 두 DataFrame을 공통 열만 포함하도록 결합
df_4 = pd.concat([df_1, df_2], join='inner')
print(df_4)

# merge를 이용한 조인 연산
# left outer join: df_1 기준
print('left outer \n', df_1.merge(df_2, how='left', on='B'))
# right outer join: df_2 기준
print('right outer \n', df_1.merge(df_2, how='right', on='B'))
# full outer join: df_1과 df_2의 모든 데이터를 포함
print('full outer \n', df_1.merge(df_2, how='outer', on='B'))
# inner join: 공통된 데이터만 포함
print('inner \n', df_1.merge(df_2, how='inner', on='B'))
