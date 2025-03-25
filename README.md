# Artificial-Life-and-Foundation-Models

A group project for COMP0258: Open-Endedness and General Intelligence, a course at UCL led by Professor Tim Rocktäschel.

"Open-endedness (OE) is the ability of a system to keep generating interesting artifacts forever, which is one of the defining features of natural evolution [Stanley et al., 2017]."

- Resources we used for inspiration and reviewed: https://docs.google.com/document/d/1GNeawJdXXLfGvWVEroBYWnJFeIx6geIDxgJdd3e7O1c/edit?tab=t.0

- Assignment Instructions: https://docs.google.com/document/d/1KWJ4eVg3fOkxuVOcFGFFyC3k4vTbnthGplZYqAAAIlg/edit?tab=t.0

## How To Run Different Parts of Code

Install everything you need with one script:
```sh
bash setup.sh
```
You will then be prompted with some further instructions, to activate the environment and to add your Hugging Face details.

### Running Python Scripts

To run different python scripts one would do:
```sh
python SCRIPT.py
```

## Environment set up

```sh
conda create -n asal python=3.10 -y && \
conda activate asal && \
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y && \
conda install numpy imageio -c conda-forge -y && \
pip install imageio[ffmpeg] && \
pip install evotorch && \
pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3
```
