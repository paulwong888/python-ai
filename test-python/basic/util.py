import datetime as dt
import csv

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