import json
# import numpy

with open("people.json", 'r', encoding='utf-8') as f:
    people =  json.load(f)
    print(type(people))
    for p in people:
        print(type(p))

help(json)