import numpy as np

l = [1,3,5,7,9]
arr = np.array(l) #conver list to numpy array

arr1 = np.ones(10)
print(arr1)

arr2 = np.zeros(10)
print(arr2)

arr3 = np.full(shape=[2,3], fill_value=2.718)
print(arr3)

arr4 = np.arange(start=0, stop=20, step=2)
print(arr)

arr5 = np.linspace(start=0, stop=20, num=10)
print(arr)

arr6 = np.random.randint(0,100, size=10)
print(arr)