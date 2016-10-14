import math,random, numpy as np

# relation to mean error and sample size?
# relation to std and sample size?
def stats(steps):
    p = 0.3
    dt = []
    for t in range(10):
        c = []
        for i in range(steps):
            d = 1 if random.random()<=p else 0
            c.append(d)
        dt.append(float(sum(c))/len(c))

    print("n: {0}".format(steps))
    print(dt)
    mean = np.mean(dt)
    print("mean: {0}".format(mean))
    std = np.std(dt)
    print("std: {0} [{1},{2}]".format(std,p-std,p+std))
    print([dat < p-std or dat > p+std for dat in dt])
    ste = std/math.sqrt(steps)
    print("ste: {0} [{1},{2}]".format(ste,p-ste,p+ste))

# stats(10)
# stats(100)
# stats(1000)

def bayes():
    p_unknown = 0.3
    p_h_d = 0.5
    count = 0
