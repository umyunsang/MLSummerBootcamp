# 구구단 프로그램

while True:
    dan = int(input("몇단까지 출력할까요(2~9단,나머지 정수 값 입력시 종료됨)\n>> "))
    if dan in range(2, 10):
        for i in range(1, 10):
            for j in range(2, dan + 1):
                if j < 6:
                    print(f"{j} X {i} = {i * j}", end="\t")
            print()
        print()

        if dan >= 6:
            for i in range(1, 10):
                for j in range(6, dan + 1):
                    print(f"{j} X {i} = {i * j}", end="\t")
                print()
    else:
        print("이용해 주셔서 감사합니다.\n개발자 : 엄윤상 (1705817)")
        break
