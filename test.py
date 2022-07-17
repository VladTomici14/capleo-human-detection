
n = int(input("n: "))
m = int(input("m: "))

while(m != 0):
    r = n % m
    n = m
    m = r

print(n)