class Student:
    def __init__(self, name, kor, eng, mat):
        self.name = name
        self.kor = kor
        self.eng = eng
        self.mat = mat
        self.total = self.kor + self.eng + self.mat
        self.avg = self.total / 3

    def get_grade(self):
        if self.avg >= 95:
            return 'A+'
        elif self.avg >= 90:
            return 'A0'
        elif self.avg >= 85:
            return 'B+'
        elif self.avg >= 80:
            return 'B0'
        elif self.avg >= 75:
            return 'C+'
        elif self.avg >= 70:
            return 'C0'
        elif self.avg >= 60:
            return 'D0'
        else:
            return 'F0'


class Classroom:
    def __init__(self):
        self.members = []

    def add_student(self, name, kor, eng, mat):
        student = Student(name, kor, eng, mat)
        self.members.append(student)

    def get_info(self):
        return [(student.name, student.kor, student.eng, student.mat, student.total, student.avg, student.get_grade()) for student in
                self.members]


classroom = Classroom()
print("성한고등학교 1학년 1반 2학기 학생성적 입력")
while True:
    print("===========================================================")
    name = input("이름: ")
    kor = int(input("국어 점수: "))
    eng = int(input("영어 점수: "))
    mat = int(input("수학 점수: "))
    classroom.add_student(name, kor, eng, mat)

    msg = input("다른 학생의 정보를 추가 하시겠습니까?(y/n): ")
    if msg == 'n':
        break

info = classroom.get_info()
print("===========================================================")
print("성한고등학교 1학년 1반 이번학기 성적 현황")
print(f'총 학생수 : {len(info)}')
print("===========================================================")
print("순번      이름       국어     영어    수학    총점   평균   학점")
print("===========================================================")
i, kor_total, eng_total, mat_total = 1, 0, 0, 0
for name, kor, eng, mat, total, avg, grade in info:
    print(f"{i}       {name}       {kor}      {eng}     {mat}     {total}    {avg:.0f}    {grade}")
    i += 1
    kor_total += kor
    eng_total += eng
    mat_total += mat
print("===========================================================")
print(f"과목총점:            {kor_total}     {eng_total}    {mat_total}  ")
print(
    f"과목평균:            {kor_total / len(info):.0f}      {eng_total / len(info):.0f}     {mat_total / len(info):.0f}  ")
print("===========================================================")
