import datetime as dt

class Member:
    expiry_days = 365
    """ Default number of free days"""
    
    """Create a new member"""
    def __init__(self,first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name
        self.date_joined = dt.date.today()
        self.expiry_date = self.date_joined + dt.timedelta(days=self.expiry_days)

paul = Member("Paul", "Wong")

print(paul.date_joined)
print(paul.expiry_date)

class Admin(Member):
    expiry_days = 365.2442 *100

    def __init__(self, first_name, last_name, secret_code):
        super().__init__(first_name, last_name)
        self.secret_code = secret_code

class User(Member):
    pass


ann = Admin("Paul", "Wong", "12345")
print(ann.first_name, ann.last_name, ann.expiry_date, ann.secret_code)