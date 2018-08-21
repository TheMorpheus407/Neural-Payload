import matplotlib.pyplot as plt
import json

losses_read = json.loads(open("losses_neural_evaluator.txt").read())
losses = []
for i in losses_read:
    for j in i:
        losses.append(j)
losses_avg = []
sum = 0
for i, item in enumerate(losses):
    sum += item
    if i % 10000 == 0:
        losses_avg.append(sum/10000)
        sum = 0
plt.figure()
plt.plot(losses_avg)
plt.show()