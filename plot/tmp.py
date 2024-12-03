import numpy as np

import matplotlib.pyplot as plt

# P8

# 64 8 -- 0.4337
# 32 8 -- 0.3505
# 16 8 -- 0.2458

# 32 32 -- 0.393
# 32 16 -- 0.3705
# 32 4 -- 0.3236


# P 16

# 64 8 -- 0.3545
# 32 8 -- 0.2904
# 16 8 -- 0.2149

# 32 4 --0.2703
# 32 8 -- 0.2919
# 32 16 --0.3015
# 32 32 -- 0.3189




depth = [16, 32, 64]
Acc = [0.2458, 0.3505, 0.4337]

plt.plot(depth, Acc, label="patch size = 8")

depth = [16, 32, 64]
Acc = [0.2149, 0.2919, 0.3545]

plt.plot(depth, Acc, label="patch size = 16")
plt.xlabel("dim")
plt.ylabel("accuracy")
plt.legend()
plt.show()





# depth = [4, 8, 16, 32]
# Acc = [0.3236, 0.3505, 0.3705, 0.393]

# plt.plot(depth, Acc, label="patch size = 8")

# depth = [4, 8, 16, 32]
# Acc = [0.2703, 0.2919, 0.3015, 0.3189]

# plt.plot(depth, Acc, label="patch size = 16")
# plt.xlabel("depth")
# plt.ylabel("accuracy")
# plt.legend()
# plt.show()


