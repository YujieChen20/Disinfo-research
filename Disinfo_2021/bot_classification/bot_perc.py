import pandas as pd
from datetime import timedelta, date
import os
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "probabilities", "probabilities.csv"))

# bot percentage
#myList = list(sorted(list(probabilities.values), reverse=False))

dic = {}
for i in range(len(df)):
    if df["user_id"][i] not in dic:
        dic[df["user_id"][i]] = []
    dic[df["user_id"][i]].append(df["bot_probability"][i])

threshold = [i/100 for i in range(51, 91, 1)]
bots_perc = []
for thre in threshold:
    bots = 0
    for probs in dic.values():
        for prob in probs:
            if prob >= thre:
                bots += 1
                break
    bots_perc.append((bots/len(dic))*100)
#print(bots_perc)


# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
plt.plot(threshold, bots_perc)
#plt.xticks([i/100 for i in range(51, 91, 5)])
#plt.yticks([i/100 for i in range(0, 101, 5)])
plt.xlabel("Threshold")
plt.ylabel("Percentage of Bot (%)")
plt.title('Percentage of Bot under Different Threshold')
#plt.show()
plt.savefig(os.path.join(os.path.dirname(__file__), "bot_percentage.png"))