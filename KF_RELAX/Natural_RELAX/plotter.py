import matplotlib.pyplot as plt
import pickle



with open ('kf_rlx', 'rb') as fp:
    kf_rlx = pickle.load(fp)

with open ('rlx', 'rb') as fp:
    rlx = pickle.load(fp)

print(kf_rlx)
p1 = plt.plot(range(len(kf_rlx)), kf_rlx, label="KF-RELAX")
p2 = plt.plot(range(len(kf_rlx)), rlx, label="RELAX")
plt.legend(["KF-RELAX", "RELAX"])
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Loss', labelpad = -3, fontsize=12)
#plt.show()
plt.savefig('toy.png')