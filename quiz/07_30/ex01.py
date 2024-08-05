name = input("당신의 이름은 : ")
w = float(input("당신의 몸무게(kg) : "))
t = float(input("당신의 키(m) : "))
bmi = w / t ** 2
print(name, "님의 bmi는", bmi, "입니다.")

# if bmi >= 30:
#     print("비만")
# else:
#     print("비만아님")

print("당신은", end=' ')
if bmi >= 39:
    print("고도 비만")
elif bmi >= 32:
    print("중도 비만")
elif bmi >= 30:
    print("경도 비만")
elif bmi >= 24:
    print("과체중")
elif bmi >= 10:
    print("정상")
else:
    print("저체중")
