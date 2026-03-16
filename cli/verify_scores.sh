source ./cli/utils.sh
activate_conda_env

python full-open-llm-leaderboard/verify_scores.py --k 5 --seed 42

