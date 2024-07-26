file_is_open = False

class CustomException(Exception):
    def __init__(self, message="A custom exception occurred"):
        super().__init__(message)

try:
    the_file = open("people.csv")
    line_count = len(the_file.readlines())
    if line_count < 2:
        raise CustomException("Not enough rows")
except FileNotFoundError:
    print("\nThere is no people.csv file here")

except Exception as e:
    print(e)
    
else:
    file_is_open = True
    
    the_file.seek(0)
    for one_line in the_file:
        print(one_line)

    the_file.seek(0)
    file_content = the_file.read()
    print(file_content)
    
finally:
    if file_is_open:
        the_file.close()
    print("File close now")
