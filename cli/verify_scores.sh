source ./cli/utils.sh
activate_conda_env

python partial-open-llm-leaderboard/verify_scores.py --benchmarks bbh gpqa --n-models 5

