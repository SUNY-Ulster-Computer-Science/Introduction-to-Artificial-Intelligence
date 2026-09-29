"""A showcase of basic Python features."""

from typing import Callable


# Variable assignment

name = "Alice"
age = 21
gpa = 3.85
is_enrolled = True
 
print(type(age))  # <class 'int'>
print(type(gpa))  # <class 'float'>
 
print(f"{name} is {age} years old with a GPA of {gpa:.1f}")  # GPA rounded to 1 decimal place

name = True  # Valid, but not recommended
print(name)

# print("hello" + 5)  # Will cause a TypeError


# Collections

# Lists
scores = [88, 72, 95, 61, 84]
 
print(scores[0])  # 88
print(scores[-1])  # 84
print(scores[1:4])  # [72, 95, 61]
 
scores.append(90)

print(scores)  # [88, 72, 95, 61, 84, 90]

# list comprehension
passing = [s for s in scores if s >= 70]
print(passing)  # [88, 72, 95, 84, 90]

# Tuples
point = (1.5, 2.5)
print(point)

# Dicts
student = {
    "name": "Alice",
    "age": 21,
    "scores": scores,
}
 
print(student["name"])  # Alice


# Control Flow

# for loop
for score in scores:
    print(score, end=' ')
print()
 
# while loop
i = 0
while i < len(scores):
    print(scores[i], end=' ')
    i += 1
print()
 
# Conditional
if "hello" == "world":
    print("if")
elif True == 1:
    print("elif")  # elif
else:
    print("else")


# Functions

def is_even(num: int) -> bool:
    return num % 2 == 0

print(is_even(3))  # False

def fire_callback(callback: Callable[[int], None]):
    callback(3)  # Executes the passed callback function

fire_callback(lambda x: print("Callback! " * x))


# Classes

class Faz:
    def __init__(self, hars: int):
        self.hars = hars

    def toreador(self):
        print("har " * self.hars)

    @classmethod
    def party(cls):
        print("Let's eat!")

    @staticmethod
    def out_of_order():
        raise RuntimeError("Sorry")

    def __str__(self):
        return "Stringified!"


f = Faz(5)
f.toreador()

print(str(f))
