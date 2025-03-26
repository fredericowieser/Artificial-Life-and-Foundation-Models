# Artificial-Life-and-Foundation-Models

A group project for COMP0258: Open-Endedness and General Intelligence, a course at UCL led by Professor Tim Rocktäschel.

"Open-endedness (OE) is the ability of a system to keep generating interesting artifacts forever, which is one of the defining features of natural evolution [Stanley et al., 2017]."

- Resources we used for inspiration and reviewed: https://docs.google.com/document/d/1GNeawJdXXLfGvWVEroBYWnJFeIx6geIDxgJdd3e7O1c/edit?tab=t.0

- Assignment Instructions: https://docs.google.com/document/d/1KWJ4eVg3fOkxuVOcFGFFyC3k4vTbnthGplZYqAAAIlg/edit?tab=t.0

## Details About The Project
### Original Code and Previous Work
In this code we look to build on some of the algorithms and research that was originally proposed in the paper "Automating the Search for Artificial Life with Foundation Models" ([Website](https://pub.sakana.ai/asal/)) by Akarsh Kumar, Chris Lu, Louis Kirsch, Yujin Tang, Kenneth O. Stanley, Phillip Isola, and David Ha.

```
@article{kumar2024asal,
  title = {Automating the Search for Artificial Life with Foundation Models},
  author = {Akarsh Kumar and Chris Lu and Louis Kirsch and Yujin Tang and Kenneth O. Stanley and Phillip Isola and David Ha},
  year = {2024},
  url = {https://asal.sakana.ai/}
}
```

### This Code
In this project we have code which relates to our group's work that uses a fork from the original code ([Original](https://github.com/SakanaAI/asal) | [Fork](https://github.com/hannaherlebach/asal/commits/main/)). This fork is a version of the original code that contains new ALife substrates, such as FlowLenia, and some other ideas that build quite nicely into the original code made using JAX.

In order to use a larger and more diverse set of Foundation Models we have started developing a PyTorch version of the original code which is stored in `asal_pytorch` directory. For now we have only implemented the Lenia substrate and are using `evotorch` ([DOCS](https://docs.evotorch.ai/v0.5.1/)) and the `transformers` library in order to both use the wider range of foundation models available through hugging face and reduce overhead in casting JAX arrays to PyTorch tensors.


## How To Run Different Parts of Code

Install everything you need with one script:
```sh
bash setup.sh
```
You will then be prompted with some further instructions, to activate the environment and to add your Hugging Face details.

If you want to add any packages simply

```sh
deactivate
```

Then add your package and a version if needed to the requirements.txt file. Then do...

```sh
bash setup.sh
```

and follow the notes again.

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
