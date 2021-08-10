# LanceOTron CLI

[![CircleCI](https://circleci.com/gh/Chris1221/lanceotron/tree/main.svg?style=svg&circle-token=bf3f78a54437e63368f5b9dc1c536d7f32f32393)](https://circleci.com/gh/Chris1221/lanceotron/tree/main)

A bare-bones interface to the trained LanceOTron (LoT) model from the command line.

## Installation

1. Clone the repository.
2. Install dependencies with pip.
3. Install the package.
4. Run tests to ensure that everything is working.

```{sh}
git clone git@github.com:Chris1221/lanceotron.git; cd lanceotron # Step 1
pip install -r requirements.txt # Step 2
pip install -e . # Step 3
python -m unittest
```

## Usage

To see available commands, use the `--help` flag.

```
lanceotron --help
```

## Call Peaks

To call peaks from a bigWig track, use the `callPeaks` command.

| Option          | Description                                            | Default |
|-----------------|--------------------------------------------------------|---------|
| file            | BigWig Track to analyse                                |         |
| -t, --threshold | Threshold for selecting candidate peaks                | 4       |
| -w, --window    | Window size for rolling mean to select candidate peaks | 400     |
| -f, --folder    | Output folder                                          | "./"    |
| --skipheader    | Skip writing the header                                | False   |


