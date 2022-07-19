import math

n = float(input("enter a float: '"))
decimals = int(input("number of decimals: "))

n = int(n * math.pow(10, decimals))
n = float(n / math.pow(10, decimals))

print(n)