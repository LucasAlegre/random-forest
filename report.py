import os

files = [('wdbc.csv','diagnosis'), ('wine.csv', 'class'), ('car.csv', 'class'), ('ionosphere.csv', 'g/b'), ('pima.tsv','target')]

for f in files:
    for n in [1, 5, 10, 25, 50]:
        cmd = "python3 random_forest.py -s 42 -n " + str(n) + " -d datasets/" + f[0] + " -c " + f[1]
        output = "results/" + f[0] + "_n_" + str(n) + '.csv'
        if f[0] == 'wdbc.csv':
            cmd += " -drop id"
        if f[0] == 'pima.tsv':
            cmd += ' -sep \"\\t\" '
        cmd += " > " + output
        print(cmd)
        os.system(cmd)