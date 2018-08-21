import json
import random
import string

with open("website_attacks.txt") as f:
    all_attacks = json.loads(f.read())

successful = 0
unsuccessful = 0
for i in all_attacks:
    if i["target"] == "-":
        unsuccessful = unsuccessful + 1
    else:
        successful = successful + 1
print(successful)
print(unsuccessful)
exit()
difference = successful - unsuccessful
negativepayloads = open("negativepayloads", "w")
x = []
for i in range(difference):
    N = random.randint(1, 30)
    s = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=N))
    x.append(s)
for i in x:
    negativepayloads.write(i + "\n")