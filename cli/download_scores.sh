#!/bin/bash

source ./cli/utils.sh
activate_conda_env

python open-llm-leaderboard-v2/download_model_scores.py
