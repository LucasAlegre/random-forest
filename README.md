# Random Forest

Implementação do método ensemble de Florestas Aleatórias para a disciplina INF01017 Aprendizado de Máquina – 2019/1.

Para documentação detalhada, consulte o arquivo [Relatório.pdf](https://github.com/LucasAlegre/random-forest/blob/master/Relatório.pdf).

## Install Requirements

```
pip3 install -r requirements.txt
```

## Run

```
python3 random_forest.py [-h] [-s SEED] [-d DATA] [-c CLASS_COLUMN] [-sep SEP]
                        [-n NUM_TREES] [-k NUM_FOLDS] [-drop DROP [DROP ...]]
                        [-not-sample] [-cut-by-mean] [-v]

Random Forest - Aprendizado de Máquina 2019/1 UFRGS

optional arguments:
  -h, --help            show this help message and exit
  -s SEED               The random seed. (default: None)
  -d DATA               The dataset .csv file. (default: datasets/wine.csv)
  -c CLASS_COLUMN       The column of the .csv to be predicted. (default:
                        class)
  -sep SEP              .csv separator. (default: ,)
  -n NUM_TREES          The number of trees in the random forest. (default: 5)
  -k NUM_FOLDS          The number of folds used on cross validation.
                        (default: 10)
  -drop DROP [DROP ...]
                        Columns to drop from .csv. (default: [])
  -not-sample           Do not sample attributes on each node. (default:
                        False)
  -cut-by-mean          Cut point by mean of numerical attribute. (default:
                        False)
  -v                    View random tree image. (default: False)
```

## Authors

* **Lucas Alegre** - [LucasAlegre](https://github.com/LucasAlegre)
* **Bruno Santana** - [bsmlima](https://github.com/bsmlima)
* **Pedro Perrone** - [pedroperrone](https://github.com/pedroperrone)



