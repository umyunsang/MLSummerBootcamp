import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 데이터프레임을 생성하기 위한 CSV 파일 읽기
path = 'https://raw.githubusercontent.com/umyunsang/MLSummerBootcamp/master/vehicle_prod1.csv'

# CSV 파일을 읽어 데이터프레임 생성
df = pd.read_csv(path)
print(df)  # 데이터프레임 전체 출력

# 첫 번째 열을 인덱스로 설정하여 데이터프레임 재생성
df = pd.read_csv(path, index_col=0)
print(df)  # 인덱스가 설정된 데이터프레임 전체 출력

# 열과 인덱스 확인
print(df.columns)  # 데이터프레임의 열 이름 출력
print(df.index)    # 데이터프레임의 인덱스 출력

# 특정 년도 데이터 선택
print(df['2007'])  # 2007년 데이터 출력

# 열과 행을 리스트로 변환
print(df.columns.tolist())  # 열 이름을 리스트로 변환하여 출력
print(df['2008'].tolist())  # 2008년 데이터를 리스트로 변환하여 출력

# 새로운 열 생성: 총합과 평균
df['total'] = df[['2007', '2008', '2009', '2010', '2011']].sum(axis=1)
df['mean'] = df[['2007', '2008', '2009', '2010', '2011']].mean(axis=1)
print(df)  # 새로운 열이 추가된 데이터프레임 출력

# 열 삭제 및 데이터프레임 갱신
print(df.drop('2007', axis=1))  # '2007' 열을 삭제한 데이터프레임 출력
df.drop('2007', axis=1, inplace=True)  # '2007' 열 삭제 (inplace로 갱신)
df['total'] = df[['2008', '2009', '2010', '2011']].sum(axis=1)  # 총합 열 재계산
df['mean'] = df[['2008', '2009', '2010', '2011']].mean(axis=1)   # 평균 열 재계산
print(df)  # 열 삭제 및 갱신 후 데이터프레임 출력

# 데이터 시각화
# 바 차트
bar = df['2009'].plot(kind='bar', color=('orange', 'r', 'b', 'c', 'k'))
plt.show()  # 바 차트 출력

# 파이 차트
pie = df['2009'].plot(kind='pie')
plt.show()  # 파이 차트 출력

# 선 차트
line = df.plot(kind='line')
plt.show()  # 선 차트 출력

# 슬라이싱과 인덱싱
print(df.head())  # 상위 5행 출력
print(df[2:6])  # 2번째부터 6번째 행까지 출력
print(df.loc['Korea'])  # 'Korea' 인덱스의 데이터 출력
print(df.loc[['US', 'Korea']])  # 'US'와 'Korea' 인덱스의 데이터 출력

# 특정 값 접근
# print(df['2011'][[0, 4]])  # 주석 처리된 코드, 특정 열의 특정 행 값 접근
print(df.loc['Korea', '2011'])  # 'Korea' 인덱스와 '2011' 열의 값 접근

# loc, iloc 인덱서 사용
print(df.iloc[4])  # 5번째 행 출력 (정수 위치 기준)
print(df.head(2)['2009'])  # 상위 2행의 '2009' 열 데이터 출력
print(df.iloc[[2, 4]])  # 3번째와 5번째 행 출력 (정수 위치 기준)
