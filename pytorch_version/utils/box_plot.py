import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

key, value = [], []
import pandas as pd

df_ERM = pd.read_excel('Density Softmax.xlsx', "ERM")
df_DSM= pd.read_excel('Density Softmax.xlsx', "VAE_calib")

def get_accuracies(df, dname):
    out = []
    for i in range(1, 41, 4):
        out.append(df[dname][i])
    return out

# dname = "iid"
# dname = "shot_noise"
# dname = "translate"
dname = "stripe"
key += ["ERM"] * 10
value += get_accuracies(df_ERM, dname)
key += ["DSM"] * 10
value += get_accuracies(df_DSM, dname)

d = {'key': key, 'value': value}
df = pd.DataFrame(data=d)

sns.boxplot(x=df["key"], y=df["value"])

plt.xlabel("Method")
plt.ylabel('ECE')

plt.title(dname)
plt.savefig(dname + "_ece.png")