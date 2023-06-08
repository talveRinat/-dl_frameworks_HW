Initialize DVC in my project directory:
```bash
dvc init
```

```bash
 dvc config core.autostage true
```

Set up DVC remote storage:
```bash
dvc remote add -d myremote /path/to/remote/storage
```

Track data file using DVC:
```bash
dvc add data.csv
```

Commit the changes to DVC:
```bash
git add data.csv.dvc
git commit -m "Add data file"
```

Track my model file using DVC:
```bash
dvc add model.joblib
```

Commit the changes to DVC:
```bash
git add model.joblib.dvc
git commit -m "Add model file"
```

poetry create requirements.txt
```bash
poetry export --without-hashes --format=requirements.txt > requirements.txt
```