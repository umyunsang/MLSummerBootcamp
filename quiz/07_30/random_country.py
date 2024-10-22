import random

countries = {
    "대한민국": "서울",
    "멕시코": "멕시코시티",
    "스페인": "마드리드",
    "프랑스": "파리",
    "영국": "런던",
    "이탈리아": "로마",
    "그리스": "아테네",
    "독일": "베를린",
    "일본": "도쿄",
    "중국": "베이징",
    "러시아": "모스크바"
}


def main_menu():
    print("=======================")
    print("수도 맞추기 게임")
    print("=======================")
    print("1. 게임시작")
    print("2. 정보 확인")
    print("3. 정보 추가")
    print("4. 정보 수정")
    print("5. 종료")


def menu_1():
    print("=======================")
    while True:
        country = random.choice(list(countries.keys()))
        answer = input(f"{country}의 수도는? ")
        if answer == "그만":
            break
        elif answer == countries[country]:
            print("정답!!")
        else:
            print("아닙니다!!")


def menu_2():
    print("=======================")
    print("지정된 나라/수도 리스트")
    print("=======================")
    for country, city in countries.items():
        print(f"{country} -> {city}")
    print("=======================")
    print(f"현재 총 {len(countries)}개의 수도가 입력되어 있습니다.")
    print("=======================")


def menu_3():
    print("=======================")
    country = input("추가할 나라 입력 > ")
    if country in countries:
        print(f"{country}는 이미 있습니다.")
    else:
        city = input("추가할 수도(도시) 입력 > ")
        countries[country] = city
        print(f"{country} -> {city} 추가되었습니다.")


def menu_4():
    print("=======================")
    choice = input("수정은 1, 삭제는 2를 입력해주세요 : ")
    if choice == '1':
        country = input("수정할 나라를 입력하세요 : ")
        if country in countries:
            city = input("수정할 수도(도시) 이름을 입력해주세요 : ")
            countries[country] = city
            print(f"{country}의 수도가 {city}(으)로 수정되었습니다.")
        else:
            print("등록된 나라가 아닙니다.")
    elif choice == '2':
        country = input("삭제할 나라를 입력하세요 : ")
        if country in countries:
            del countries[country]
            print(f"{country} 삭제가 완료되었습니다.")
        else:
            print("등록된 나라가 아닙니다.")
    else:
        print("잘못된 입력입니다.")


while True:
    main_menu()
    choice = input(">> ")
    if choice == '1':
        menu_1()
    elif choice == '2':
        menu_2()
    elif choice == '3':
        menu_3()
    elif choice == '4':
        menu_4()
    elif choice == '5':
        print("프로그램을 종료합니다.")
        break
    else:
        print("잘못된 입력입니다. 다시 시도해주세요.")
