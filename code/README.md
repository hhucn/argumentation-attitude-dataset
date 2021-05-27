## Setup

```
virtualenv --python=python3.8 env
source env/bin/activate
pip3 install -r requirements.txt
```

## Run

```
python3 main.py [conviction,strength,precision@3] [majority,deliberate] /path/to/arguments.csv /path/to/train.csv /path/to/test.csv [lowest-eval-statement-id] [highest-eval-statement-id]
```

e.g.
```
python3 main.py precision@3 deliberate /tmp/argumentation-attitude-dataset-paper/arguments.csv /tmp/argumentation-attitude-dataset-paper/T2_T3/train.csv /tmp/argumentation-attitude-dataset-paper/T2_T3/test.csv 325 362
```

Use `config.py` to set parameters for parameter search.
