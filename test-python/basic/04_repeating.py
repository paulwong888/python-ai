for x in range(7):
    print(x)
print("All done")

print("------------")
for x in "Paul":
   print(x)
print("All done") 

print("------------")
answers = ["A", "C", "", "D"]
for answer in answers:
    if answer == "":
        print("Incomplete")
        break
    print(answer)
print("Loop is done")

print("------------")
for answer in answers:
    if answer == "":
        print("Incomplete")
        continue
    print(answer)
print("Loop is done")

print("------------")
counter = 65
while counter < 91:
    print(str(counter) + "=" + chr(counter))
    counter += 1
print("All done")