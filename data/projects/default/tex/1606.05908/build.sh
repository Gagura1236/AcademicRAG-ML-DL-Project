#!/bin/bash
pdflatex vae_tutorial.tex
bibtex vae_tutorial.aux
pdflatex vae_tutorial.tex
pdflatex vae_tutorial.tex
