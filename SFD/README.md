# SFD

The code is maintained in https://github.com/zhb2000/SFD . Please check the repository for the most up-to-date version.

# Run
## Configuration

Experiment parameters can be configured in the `.py` scripts located in the `scripts` directory. Some other configurations are in the `src/global_config.py`.

To run these scripts, ensure your current working directory is the project root. Then, execute the scripts using the following command.

## Preprocessing

Note: Set the `PYTHONPATH` to the `src` directory so that the modules can be imported correctly.

1. Convert a balanced dataset to a long-tailed dataset:

```bash
PYTHONPATH=src python scripts/make_lt_dataset.py
```

2. Split the long-tailed dataset as a federated long-tailed dataset:

```bash
PYTHONPATH=src python scripts/make_fedlt_dataset.py
```

## Training and Testing

Run training:

```bash
PYTHONPATH=src python scripts/train_sfd.py
```

Run testing:

```bash
PYTHONPATH=src python scripts/test.py
```
