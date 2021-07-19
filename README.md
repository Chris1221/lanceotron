# lanceotron

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


## To install

```sh
conda create -n lanceotron python=3.8; conda activate lanceotron
pip install -r requirements
pip install -e . 

lanceotron -h
```