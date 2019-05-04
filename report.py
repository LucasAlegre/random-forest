import os

files = [('ionosphere.csv', 'g/b'), ('wdbc.csv','diagnosis'), ('wine.csv', 'class')]

for f in files:
    for n in range(1, 31):
        for k in [5, 10]:
            for sample in [True, False]:
                for cut_by_mean in [True, False]:
                    cmd = "python3 random_forest.py -s 1234 -n " + str(n) + " -d datasets/" + f[0] + " -c " + f[1] + " -k " + str(k)
                    output = "out/" + f[0] + "_n_" + str(n) + "_k_" + str(k) 
                    if sample:
                        cmd += " -not-sample"
                        output += "_not-sample"
                    if cut_by_mean:
                        cmd += " -cut-by-mean"
                        output += "_by-mean"
                    if f[0] == 'wdbc.csv':
                        cmd += " -drop id"
                    output += ".txt"
                    cmd += " > " + output
                    print(cmd)
                    os.system(cmd)
