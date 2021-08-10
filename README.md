# LanceOTron CLI

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

Making the Lance-o-tron model slightly easier to use through a python package.

Main improvements:

- Model files are stored internally with the use of `pkg_resources`. They can still be specified on the command line if desired.
- A centralised CLI dispatch with a single entry point `lanceotron`
- Small improvements to usability
    - Hiding uninformative warning messages
    - Progress bars 

Working so far:
- `lanceotron_genome.py` is now `lanceotron callPeak` with the same parameters
- `lanceotron_scoreBed.py` is now `lanceotron scoreBed` with the same parameters.

I haven't converted the other scripts over yet. 


## To install3
n

```sh
conda create -n lanceotron python=3.8; conda activate lanceotron
pip install -r requirements
pip install -e . 

lanceotron -h
```