source ./cli/utils.sh
activate_conda_env

python open-llm-leaderboard-v2/check_model_scores.py --k 10
