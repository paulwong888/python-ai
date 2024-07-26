grades = ["C", "B", "A", "D", "C", "B", "C"]
b_index = grades.index("B")

print("The first B is index " + str(b_index))

prices = (29.95, 9.98, 4.95, 79.98, 2.95)
print(len(prices))

sample_set = {1.98, 98.9, 74.95, 2.5, 1, 16.3}
print(sample_set)
print(74.95 in sample_set)

sample_set.add(11.23)
sample_set.update([88, 123.45, 2.98])

print(sample_set)

people = {
    'htanaka': 'Haru Tanaka',
    'ppatel': 'Priya Patel',
    'bagarcia': 'Benjamin Alberto Garcia',
    'zmin': 'Zhang Min',
    'afarooqi': 'Ayesha Farooqi',
    'hajackson': 'Hanna Jackson',
    'papatel': 'Pratyush Aarav Patel',
    'hrjackson': 'Henry Jackson'
}
print(people['zmin'])
print("hajackson" in people)
print(people.get("bagarcia"))

people["hajackson"] = "Hanna Jackson-Smith"
print(people["hajackson"])

people.update({"wwiggins" : "Wanda Wiggins"})
print(people)

for person in people.keys():
    print(person + " = " + people[person])