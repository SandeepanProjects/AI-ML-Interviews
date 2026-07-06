a = [1, 2, 2, 3, 4, 4, 5]
res = []

for x in a:
    if x not in res:
        res.append(x)

print(res)


Using List Comprehension

b = [1, 2, 2, 3, 4, 4, 5]
res = []
[res.append(x) for x in b if x not in res]
print(res)

