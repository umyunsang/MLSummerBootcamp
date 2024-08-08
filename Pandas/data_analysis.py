import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# CSV 파일 경로
path = 'https://raw.githubusercontent.com/umyunsang/MLSummerBootcamp/master/weather1.csv'

# CSV 파일을 읽어와 DataFrame 생성
weather = pd.read_csv(path, index_col=0)  # 첫 번째 열을 인덱스로 설정
print(weather.head())  # 데이터프레임의 상위 5행 출력
print(weather.tail())  # 데이터프레임의 하위 5행 출력
print(weather.shape)  # 데이터프레임의 행과 열의 수 출력

# 판다스를 이용한 데이터 분석
print(weather.describe())  # 수치형 데이터의 통계적 요약 출력
print(weather.mean())      # 각 열의 평균값 출력
print(weather.std())       # 각 열의 표준편차 출력

# 데이터 정제와 결손값의 처리
print(weather.count())  # 결측치가 없는 각 열의 유효 데이터 수 출력
missing_data = weather[weather['최대풍속'].isna()]  # '최대풍속' 열의 결측치가 있는 행 선택
print(missing_data)  # 결측치가 있는 데이터 출력

# 결측치를 '평균풍속' 열의 평균값으로 대체
weather.fillna(weather['평균풍속'].mean(), inplace=True)
print(weather.loc['2012-02-12'])  # 특정 날짜의 데이터 출력

# 시계열 데이터 분석
d_list = ['01/03/2018', '01/03/2018', '2018/01/05', '2018/01/06']
# 문자열 리스트를 DateTimeIndex로 변환
print(pd.DatetimeIndex(d_list).year)  # 연도 출력
print(pd.DatetimeIndex(d_list).month)  # 월 출력
print(pd.DatetimeIndex(d_list).day)    # 일 출력

# CSV 파일을 다시 읽어와 DataFrame 생성 (인덱스 설정 없이)
weather = pd.read_csv(path)
# '일시' 열을 기준으로 월 정보를 추출하여 새로운 열 'month' 추가
weather['month'] = pd.DatetimeIndex(weather['일시']).month
# 월별로 그룹화하여 각 열의 평균값 계산
means = weather.groupby('month').mean(numeric_only=True)
print(means)  # 월별 평균값 출력

# '일시' 열을 기준으로 연도 정보를 추출하여 새로운 열 'year' 추가
weather['year'] = pd.DatetimeIndex(weather['일시']).year
# 연도별로 그룹화하여 각 열의 평균값 계산
yearly_means = weather.groupby('year').mean(numeric_only=True)
print(yearly_means)  # 연도별 평균값 출력

# 연도별 '평균풍속'이 4.0 이상인지 여부를 논리값으로 출력
print(yearly_means['평균풍속'] >= 4.0)
