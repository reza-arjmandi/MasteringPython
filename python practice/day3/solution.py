n=int(input())
m=n % 2
if ((m!=0) or (n>=6 and n<=20)):
    print("Weird")
elif ((n>=2 and n<=5)or (n>20)):
    print("Not Weird")
