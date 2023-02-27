# VideoMAE

This repo runs Huggingface's implementation of [VideoMAE](https://huggingface.co/docs/transformers/model_doc/videomae).

The code is borrowed from this [tutorial](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/VideoMAE).

# Run with Moreh framrework
        conda create -n videomae python=3.8
        conda activate videomae
        update-moreh --firce --targer 23.3.0
        pip install ipywidgets

# Inference
        python videomae.py