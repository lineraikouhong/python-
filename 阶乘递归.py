def p(n):
    if n==1 or n==0:
        return 1
    else:
        return n*p(n-1)
print(p(3))
