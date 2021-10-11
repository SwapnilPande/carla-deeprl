import numpy as np
import matplotlib.pyplot as plt

ent_coeff = np.array([
    0.0001, 0.001, 0.0005, 0.01, 0.05, 0.1, 0.5, 1, 5
])


rewards = np.array([
    -80.56855151653289,
    -16.996567003655947,
    -16.72525464597711,
    -17.0915127280241,
    -60.071040126681325,
    -20.95644359946018,
    -38.99661116784991,
    -92.3767076164484,
    -91.42830503284931
])

plt.xscale('log')
plt.xlim(0.0001, 10)
plt.scatter(ent_coeff, rewards)


plt.xlabel("Entropy Coefficient (log)")
plt.ylabel("Max Reward")
plt.title("Max Reward vs. Entropy Coefficient")

plt.savefig("test.png")