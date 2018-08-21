import string
import random

with open("xss.txt") as f:
    lines = f.readlines()
for count, i in enumerate(lines):
    if count % 100 == 0:
        print("[%.3f]" % (100*count/len(lines)))
    if "SCRIPT" in i:
        lines.append(i.replace("SCRIPT", "script"))
        lines.append(i.replace("SCRIPT", "sCript"))
        lines.append(i.replace("SCRIPT", "scriPt"))
        lines.append(i.replace("SCRIPT", "scripT"))
        lines.append(i.replace("SCRIPT", "SCript"))
        lines.append(i.replace("SCRIPT", "scrIPT"))
        lines.append(i.replace("SCRIPT", "SCIipt"))
        lines.append(i.replace("SCRIPT", "sCRIPT"))
    if "XSS" in i:
        for j in range(50):
            N = random.randint(1,30)
            s = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=N))
            lines.append(i.replace("XSS", s))
    if "icconsult" in i:
        for j in range(50):
            N = random.randint(1,30)
            s = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=N))
            lines.append(i.replace("icconsult", s))
open("xss.txt", "w").writelines(lines)