#!/bin/bash

activate_conda_env() {
    source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
        || source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate evaltree
}