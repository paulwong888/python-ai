import datetime as dt
import csv
import util

"""""
def fname(any):
    try:
        nm = any.split(",")
        return nm[1]
    except IndexError:
        return ""
    
def lname(any):
    try:
        nm = any.split(",")
        return nm[0]
    except IndexError:
        return ""
    
def integer(any):
    return int(any.strip() or 0)
    # return 0 if any.strip() == "" else int(any.strip() or 0)

def date(any):
    try:
        return dt.datetime.strptime(any.strip(), "%m/%d/%Y").date()
    except ValueError:
        return None
    
def boolean(any):
    return True if any.strip() == "TRUE" else False

def floatnum(any):
    s_float = any.replace("$", "").replace(",", "").strip()
    return float(s_float or 0)
"""
    

people = []

class Person:
    def __init__(self, id, first_name, last_name, birth_year, date_joined,is_active, balance):
        self.id = id
        self.first_name = first_name
        self.last_name = last_name
        self.birth_year = birth_year
        self.date_joined = date_joined
        self.is_active = is_active
        self.balance = balance

with open("people.csv", encoding="utf-8", newline="") as f:
    reader = enumerate(csv.reader(f))
    f.readline()
    for i, row in reader:
        # print(row[0],row[1],row[2],row[3],row[4])
        people.append(Person(i, util.fname(row[0]), util.lname(row[0]), util.integer(row[1]), util.date(row[2]),
                                util.boolean(row[3]), util.floatnum(row[4])))
        
for p in people:
    print(f"{p.id} {p.first_name} {p.last_name:<15} {p.birth_year:<7} {p.date_joined} {p.is_active} {p.balance:>12}")




