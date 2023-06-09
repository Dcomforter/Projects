# Introduction to classes

class Person:
    def __init__(self, name, age, sex): #__init__() initializes the attributes of a class
        self.myname = name # self is used to assign arguments to the variables of a class
        self.myage = age
        self.mysex = sex

    def introduce(self):
        print(f"Hello everyone, my name is {self.myname}, I am {self.myage} years old, and I am a {self.mysex}.")

class Student(Person):
    def __init__(self, name, age, sex, major, graduation_year):
        super().__init__(name, age, sex) # super() is used to access the methods/properties of a parent class in a child class
        self.mymajor = major
        self.mygraduation_year = graduation_year

    def student_info(self):
        print(f"{self.myname} is studying {self.mymajor} and graduates in {self.mygraduation_year}.")

class Employee(Person):
    def __init__(self, name, age, sex, position, year_joined):
        super().__init__(name, age, sex)
        self.position = position
        self.year_joined = year_joined

    def employee_info(self):
        print(f"This is {self.myname}, he is a {self.mysex} and he is {self.myage} years old. \nHe is the new {self.position}."
              f" He recently joined us in {self.year_joined}.")

person1 = Person('Israel', 37, 'Male')
print(person1.introduce())
print()
student1 = Student('Olamide', 29, 'Female', 'Information Systems', 'May 2023')
print(student1.student_info())
employee1 = Employee("Israel Okuneye", 37, "Male", "Senior Engineer", 2023)
print()
print(employee1.employee_info())
