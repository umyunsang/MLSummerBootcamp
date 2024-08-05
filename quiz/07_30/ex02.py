name = input("당신의 이름은 : ")
kor = float(input("국어 성적 : "))
mat = float(input("수학 성적 : "))
eng = float(input("영어 성적 : "))
sum = kor + mat + eng
avg = sum/3
print(f'{name}님의 점수 합계는 {sum:.0f}점이고, 평균은 {avg:.6f} 입니다.')
if avg >= 70:
    print(f'{name}님은 이번 학기 통과 하셨습니다.')
else:
    print(f'{name}님은 계절학기를 수강신청 하셔야 통과 가능합니다.')

