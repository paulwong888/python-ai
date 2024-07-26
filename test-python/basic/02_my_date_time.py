import datetime

todate = datetime.date.today();
last_of_teens = datetime.date(2019, 12, 31);
my_time = datetime.time();
right_now = datetime.datetime.now();

print(todate)
print(last_of_teens)
print(my_time)
print(type(my_time))
print(right_now)

new_years_day = datetime.date(2024, 1, 1)
memorial_day = datetime.date(2024, 5, 27)
days_between = memorial_day - new_years_day
print(days_between)
print(type(days_between))
print(type(days_between.days))

birthdate = datetime.date(1980, 3, 24)
delta_age = (todate - birthdate)
days_old = delta_age.days

years = days_old // 365
months = (days_old % 365) // 30
print(f"You are {years} and {months} months old.")