#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Say "Hello, World!" With Python

print("Hello, World!")


#Python If-Else

#!/bin/python3

import math
import os
import random
import re
import sys


n = int(input())

if n % 2 == 0:
    if n in range(2,6):
        print("Not Weird")
    elif n in range(6,21):
        print("Weird")
    elif n > 20:
        print("Not Weird")
else:
    print("Weird")

    
#Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())

    print(a + b)
    print(a - b)
    print(a * b)


#Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())

    print(a//b)
    print(a/b)


#Loops
if __name__ == '__main__':
    n = int(input())
    i = 0
    while n>i:
        print(i**2)
        i += 1

        
#Write a function
def is_leap(year):
    leap = False
    
    if year%400 == 0:
        leap = True
    elif year%4 == 0:
        if year%100 == 0:
            leap = False
        else:
            leap = True
            
    return leap

#Print Function
if __name__ == '__main__':
    n = int(input())
    for i in range(1,n+1):
        if i <= n:
            print(i, end = "")

            
#List Comprehensions
if __name__ == '__main__':
    x = int(raw_input())
    y = int(raw_input())
    z = int(raw_input())
    n = int(raw_input())
    lst = [[x2,y2,z2] for x2 in range(x + 1) for y2 in range(y + 1) for z2 in range(z + 1) if x2 + y2 + z2 != n]
    print(lst)

    
#Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    lst = list(arr)
    lst.sort(reverse=True)

    mx = max(lst)

    while max(lst) == mx:
        lst.remove(mx)

    print(lst[0])

    
#Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    
    a = sum(student_marks[query_name])/len(student_marks[query_name])
    print("%.2f" % a)



#Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    
    a = sum(student_marks[query_name])/len(student_marks[query_name])
    print("%.2f" % a)


#Lists
if __name__ == '__main__':
    N = int(input())
    my_list = []
    while N > 0 :
        comm = input()
        comman = comm.split()   
        if comman[0] == "insert":
            i = int(comman[1])
            n = int(comman[2])
            my_list.insert(i,n)
        elif comman[0] == "print":
            print(my_list)
        elif comman[0] == "remove":
            n = int(comman[1])
            my_list.remove(n)
        elif comman[0] == "append":
            n = int(comman[1])
            my_list.append(n)
        elif comman[0] == "sort":
            my_list.sort()
        elif comman[0] == "pop":
            my_list.pop(-1)
        elif comman[0] == "reverse":
            my_list.reverse()
            
        N -= 1

#Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    lst = list(integer_list)
    def convert(my_list): 
        return tuple(i for i in my_list)
    t = convert(my_list= lst)
    print(hash(t))


#Nested Lists
if __name__ == '__main__':
    dic = {}
    for _ in range(int(input())):
        name = input()
        score = float(input())
        dic[name] = score
    
scores = dic.values()

second_score = sorted(list(set(scores)))[1]

second_list = []

for key, value in dic.items():
    
    if value == second_score:
        second_list.append(key)


second_list.sort()

for name in second_list:
    print(name)


#sWAP cASE
def swap_case(s):
    ans = ""
    for letter in s:
        if letter.islower():
            ans += letter.upper()
        else:
            ans += letter.lower()
    return ans


#String Split and Join
def split_and_join(line):
    a = line.split(" ")
    a = "-".join(a)
    return a


#What's Your Name?
def print_full_name(a, b):
    line = "Hello " + a + " " + b + "! You just delved into python."
    print(line)


#Mutations
def mutate_string(string, position, character):
    string = string[:position] + character + string[position+1:]
    return string


#Find a string
def count_substring(string, sub_string):
    c = 0
    match = 0
    for _ in range(0, len(string)):
        if string[c:(len(sub_string) + c)] == sub_string:
            match = match + 1
        c = c + 1
    return match


#String Validators
if __name__ == '__main__':
    s = input()
    alphanumeric = False
    alphabetical = False
    digits = False
    lowercase = False
    uppercase = False

    for letter in s:
        if letter.isalnum():
            alphanumeric = True
        if letter.isalpha():
            alphabetical = True
        if letter.isdigit():
            digits = True
        if letter.islower():
            lowercase = True
        if letter.isupper():
            uppercase = True
    
    print(alphanumeric)
    print(alphabetical)
    print(digits)
    print(lowercase)
    print(uppercase)


#Text Alignment
#Replace all ______ with rjust, ljust or center. 

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

    

#Text Wrap
import textwrap

def wrap(string, max_width):
    return textwrap.fill(string,max_width)

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)


#Designer Door Mat
# Enter your code here. Read input from STDIN. Print output to STDOUT

inp = input()

n, m = inp.split(" ")

n = int(n)
m = int(m)

t = n // 2
c = 0

while t > c:
    print(((2*c+1)*".|.").center(m,"-"))
    c += 1
    
print("WELCOME".center(m,"-"))

t = (n // 2) - 1
c = -1

while t > c:
    print(((2*t+1)*".|.").center(m,"-"))
    t -=1


#String Formatting
def print_formatted(number):
    width = len(bin(number)) - 2
    res = ""
    for i in range(1, number+1):
        print(str(i).rjust(width, ' '), str(oct(i)[2:]).rjust(width, ' '), str(hex(i)[2:].upper()).rjust(width, ' '), str(bin(i)[2:]).rjust(width, ' '))


#Alphabet Rangoli
def print_rangoli(size):
    import string

    letters = list(string.ascii_lowercase)
    
    lst = []
    
    for i in range(n):
        string = "-".join(letters[i:n])
        lst.append((string[::-1]+string[1:]).center(4*n-3, "-"))
    print('\n'.join(lst[:0:-1]+lst))


#Capitalize!


# Complete the solve function below.
def solve(s):
    lst = s.split(" ")
    newlst = []
    for name in lst:
        name = name.capitalize()
        newlst.append(name)
    
    return " ".join(newlst)


#The Minion Game
def minion_game(string):

    vowels = ["O","A","E","I","U"]
    kevinScore = 0
    stuartScore = 0

    c = 0

    for i in string:
        
        if i in vowels:
            kevinScore += (len(string) - c)
        else:
            stuartScore += (len(string) - c)

        c += 1
    
    if kevinScore > stuartScore:
        print("Kevin",kevinScore)
    elif kevinScore < stuartScore:
        print("Stuart",stuartScore)
    else:
        print("Draw")



#Merge the Tools!
def merge_the_tools(string, k):

    counter = 0
    newLst = []

    for item in string:

        counter += 1

        if item not in newLst:

            newLst.append(item)

        if counter == k:

            print(''.join(newLst))

            newLst = []
            counter = 0

#Introduction to Sets
def average(array):
    return sum(set(array)) / len(set(array))
    

#No Idea!
n, m = input().split()
myarray = input().split()
A, B = (input().split(), input().split())
A, B = set(A), set(B)


happiness = 0

for i in myarray:
    if i in A:
        happiness += 1
    elif i in B:
        happiness -= 1

print(happiness)


#Symmetric Difference
n, lst=(int(input()), input().split())
n2, lst2=(int(input()), input().split())

set1 = set(lst)
set2 = set(lst2)

un = set1.union(set2)
inter = set1.intersection(set2)
result = un.difference(inter)

print ('\n'.join(sorted(result, key = int, reverse = False)))


#Set .add()
n = int(input())
countries = set()

for i in range(n):
    countries.add(input())

print(len(countries))


#Set .discard(), .remove() & .pop()
n = int(input())
myset = map(int,input().split())
myset = set(myset)
commands_num = int(input())

for i in range(commands_num):
    command = input().split()
    

    if command[0] == "pop":
        myset.pop()
    elif command[0] == "remove":
        num = int(command[1])
        myset.remove(num)
    elif command[0] == "discard":
        num = int(command[1])
        myset.discard(num)

print(sum(myset))


#Set .union() Operation
n = int(input())
first = map(int,input().split())
first = set(first)
m = int(input())
second = map(int,input().split())
second = set(second)

print(len(first.union(second)))


#Set .intersection() Operation
n = int(input())
first = map(int,input().split())
first = set(first)
m = int(input())
second = map(int,input().split())
second = set(second)

print(len(first.intersection(second)))


#Set .difference() Operation
n = int(input())
first = map(int,input().split())
first = set(first)
m = int(input())
second = map(int,input().split())
second = set(second)

print(len(first.difference(second)))


#Set .symmetric_difference() Operation
n = int(input())
first = map(int,input().split())
first = set(first)
m = int(input())
second = map(int,input().split())
second = set(second)

print(len(first.symmetric_difference(second)))


#Set Mutations
len_of_a = int(input())
a = map(int,input().split())
a = set(a)

n_of_sets = int(input())

for i in range(n_of_sets):
    
    m = input().split()
    m_set = map(int,input().split())
    m_set = set(m_set)
    
    if m[0] == "update":
        a.update(m_set)
    elif m[0] == "intersection_update":
        a.intersection_update(m_set)
    elif m[0] == "difference_update":
        a.difference_update(m_set)
    elif m[0] == "symmetric_difference_update":
        a.symmetric_difference_update(m_set)
    
print(sum(a))


#The Captain's Room
k = int(input())
rooms = map(int,input().split())
rooms = list(rooms)
captain_room = set()
family_room = set()

for elment in rooms:
    if elment not in captain_room:
        captain_room.add(elment)
    else:
        family_room.add(elment)

print(sum(captain_room.difference(family_room)))


#Check Subset
number_of_tests = int(input())

for i in range(number_of_tests):
    
    n_first = int(input())
    first = map(int, input().split())
    first = set(first)
    
    n_second = int(input())
    second = map(int, input().split())
    second = set(second)
    
    temp = first.intersection(second)
    
    print(temp == first)


#Check Strict Superset
first = map(int, input().split())
first = set(first)

n_sets = int(input())

for i in range(n_sets):
    
    new_set = map(int, input().split())
    new_set = set(new_set)
    
    res = True
    
    if first.issuperset(new_set) == False:
        res = False
        break
    elif first.issuperset(new_set):
        if first == new_set:
            res = False
            break


print(res)


#collections.Counter()
number_of_shoes = int(input())
all_shoes = list(map(int, input().split()))
number_of_customers = int(input())

total_sale = 0

for elment in range(number_of_customers):
    customer = list(map(int, input().split()))
    if customer[0] in all_shoes:
        total_sale += customer[1]
        all_shoes.remove(customer[0])

print(total_sale)


#DefaultDict Tutorial

from collections import defaultdict

n, m = input().split()

m_list = []
result = defaultdict(list)

for i in range(int(n)):
    
    elment = input()
    
    result[elment].append(i + 1)
    
    
for i in range(int(m)):
    
    elment = input()
    
    m_list.append(elment)
    
for elment in m_list:
        
        if elment in result:
            
            print(" ".join(map(str, result[elment])))
        else:
            print(-1)

            
#Collections.namedtuple()
from collections import namedtuple

number_of_cases = int(input())

columns = input()

total_scores = 0

for i in range(number_of_cases):
    student = namedtuple("student", columns)
    inp = input().split()
    a = student(inp[0], inp[1], inp[2], inp[3])
    
    total_scores += int(a.MARKS)
    

avg = total_scores/number_of_cases

print("{:.2f}".format(avg))
            
            
#Collections.OrderedDict()
from collections import OrderedDict

number_of_items = int(input())

my_dict = OrderedDict()

for i in range(number_of_items):
    
    elment = input().split()
    
    price = int(elment[-1])
    
    item_name = " ".join(elment[:-1])
    
    
    if my_dict.get(item_name):
        
        my_dict[item_name] += price
        
    else:
        
        my_dict[item_name] = price

for item , price in my_dict.items():
    
    print(str(item) + " " + str(price))

    
#Word Order
from collections import Counter

n = int(input())

words = []

for i in range(n):
    
    words.append(input().strip())

my_counter = Counter(words)

print(len(my_counter))

for elment in words:
    
    result = my_counter.pop(elment, None)
    
    if result == None:
        
        continue
        
    else:
        
        print(result, end= " ")
    

#Collections.deque()
from collections import deque

n = int(input())

result = deque()

for i in range(n):
    
    command = input().split()
    
    if command[0] == "append":
        result.append(command[1])
    if command[0] == "appendleft":
        result.appendleft(command[1])
    if command[0] == "pop":
        result.pop()
    if command[0] == "popleft":
        result.popleft()


print(" ".join(result))
    

#Calendar Module
import calendar

my_day = input().split()

w_day = calendar.weekday(int(my_day[2]), int(my_day[0]), int(my_day[1]))

days = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]

print(days[w_day])


#Exceptions
n = int(input())

for i in range(n):
    try:
        numbers = list(map(int,input().split()))
        print(numbers[0] // numbers[1])
    except Exception as error:
        print("Error Code:", error)

        
#Zipped!
number_of_students , x_subjects = input().split()

mark_sheet = []

for i in range(int(x_subjects)):

    mark_sheet.append(map(float, input().split())) 

for i in zip(*mark_sheet):

    print(sum(i) / len(i))


#Athlete Sort
import math
import os
import random
import re
import sys



if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
    
for l in sorted(arr, key= lambda x: x[k]):
    print(*l)


#ginortS
my_string = input()

lower_alph = []
upper_alph = []
odd_num = []
even_num =[]


for i in sorted(my_string):
    
    if i.isalpha():
        
        if i.isupper():
            
            upper_alph.append(i)
        else:
            lower_alph.append(i)
        
    else:
        
        if int(i) % 2 == 0 :
            
            even_num.append(i)
        
        else:
            odd_num.append(i)
        

    
print("".join(lower_alph + upper_alph + odd_num + even_num))


#Map and Lambda Function
cube = lambda x: x**3 # complete the lambda function 

def fibonacci(n):
    
    if n == 0:
        
        return([])
    
    if n == 1:
        
        return([0])
    
    result = [0, 1]
    
    for i in range(2,n):
        result.append(result[i-1] + result[i-2])
    return(result)
    

#Detect Floating Point Number
import re

n = int(input())

for i in range(n):
    my_string = input()
    
    print(bool(re.match(r'^[-+]?[0-9]*\.[0-9]+$', my_string)))


#Re.split()
regex_pattern = r"[.,]+"


#Group(), Groups() & Groupdict()
import re
reg = re.match(r".*?([a-zA-Z0-9]+)\1", input())
if reg:
    print(reg.group(1))
else:
    print(-1)


#Re.findall() & Re.finditer()
import re

my_string = input()

vowels = 'aeiou'
constants = 'bcdfghjklmnpqrstvwxyz'

result = re.findall(r'(?<=[' + constants + '])([' + vowels + ']{2,})[' + constants + ']', my_string, re.IGNORECASE)

if result:
    
    print(*result, sep='\n')
    
else:
    
    print('-1')

    
#Re.start() & Re.end()
import re
S, k = input(), input()

if not re.search(k, S):
    
    print('(-1, -1)')
    
else:
    
    i = 0
    
    while re.search(k, S[i:]):
        
        i += re.search(k, S[i:]).start() + 1
        
        print('(', i-1, ', ', i+len(k)-2, ')', sep='')

        
#Re.start() & Re.end()
import re
S, k = input(), input()

if not re.search(k, S):
    
    print('(-1, -1)')
    
else:
    
    i = 0
    
    while re.search(k, S[i:]):
        
        i += re.search(k, S[i:]).start() + 1
        
        print('(', i-1, ', ', i+len(k)-2, ')', sep='')



#Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        f(["+91 "+c[-10:-5]+" "+c[-5:] for c in l])
    return fun


#Decorators 2 - Name Directory
def person_lister(f):
    def inner(people):
        return map(f, sorted(people, key=lambda x: int(x[2])))
    return inner


#Arrays
def arrays(arr):
    
    return(numpy.array(arr, float)[::-1])


#Shape and Reshape
import numpy

inp = list(map(int, input().split()))

my_array = numpy.array(inp)
print(numpy.reshape(my_array,(3,3)))


#Transpose and Flatten
import numpy

n, m = input().split()

row = []

for i in range(int(n)):
    
    my_array = input().split()
    
    row.append(my_array)
    
my_array = numpy.array(row, int)

print(numpy.transpose(my_array))

print(my_array.flatten())


#Concatenate
import numpy

n, m, p = input().split()

my_array_1 = []
my_array_2 = []

for i in range(int(n)):
    
    row1 = list(map(int, input().split()))
    my_array_1.append(row1)
    
for i in range(int(m)):
    row2 = list(map(int, input().split()))
    my_array_2.append(row2)

print(numpy.concatenate((my_array_1, my_array_2), axis = 0))


#Zeros and Ones
import numpy

inp = list(map(int, input().split()))

print(numpy.zeros(inp, dtype = numpy.int))

print(numpy.ones(inp, dtype = numpy.int))


#Eye and Identity
import numpy
print(str(numpy.eye(*map(int,input().split()))).replace('1',' 1').replace('0',' 0'))


#Array Mathematics
import numpy

n, m = map(int, input().split())

a, b = (numpy.array([input().split() for _ in range(n)], dtype=int) for _ in range(2))

print(a+b, a-b, a*b, a//b, a%b, a**b, sep='\n')


#Floor, Ceil and Rint
import numpy

numpy.set_printoptions(sign=' ')

my_array = numpy.array(input().split(),float)


print(numpy.floor(my_array))
print(numpy.ceil(my_array))
print(numpy.rint(my_array))


#Sum and Prod
import numpy

n, m = map(int, input().split())


my_array = numpy.array([input().split() for i in range(n)], int)


print(numpy.prod(numpy.sum(my_array, axis=0), axis=0))


#Min and Max
import numpy

n, m = map(int, input().split())


my_array = numpy.array([input().split() for i in range(n)], int)


print(numpy.max(numpy.min(my_array, axis=1), axis=0))



#Mean, Var, and Std
import numpy

n, m = map(int, input().split())


my_array = numpy.array([input().split() for i in range(n)], int)

numpy.set_printoptions(legacy='1.13')
print(numpy.mean(my_array, axis = 1))
print(numpy.var(my_array, axis = 0))
print(numpy.std(my_array))


#Dot and Cross
import numpy

n = int(input())


my_array_A = numpy.array([input().split() for i in range(n)], int)
my_array_B = numpy.array([input().split() for i in range(n)], int)


print(numpy.dot(my_array_A, my_array_B))


#Inner and Outer
import numpy

my_array_A = numpy.array(input().split(), int)

my_array_B = numpy.array(input().split(), int)

print(numpy.inner(my_array_A, my_array_B))
print(numpy.outer(my_array_A, my_array_B))


#Polynomials
import numpy

my_array = list(map(float,input().split()))

x = int(input())

print(numpy.polyval(my_array, x))


#Linear Algebra
import numpy

n = int(input())

my_array = numpy.array([input().split() for _ in range(n)],float)

numpy.set_printoptions(legacy='1.13')

print(numpy.linalg.det(my_array))

