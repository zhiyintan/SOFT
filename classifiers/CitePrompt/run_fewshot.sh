PYTHONPATH=python
SEED=100 # 145 146 147 148
SHOT=10 # 1 2 5 10 20
VERBALIZER=kpt #soft auto
FILTER=tfidf_filter # none
KPTWLR=0.0 # 0.06
MAXTOKENSPLIT=-1 # 1
RESULTPATH="results_fewshot"
DATASET="acl_arc"


$PYTHONPATH fewshot.py \
--result_file $RESULTPATH \
--result_file results_fewshot_norefine.txt \
--seed $SEED \
--shot $SHOT \
--verbalizer $VERBALIZER \
--max_token_split $MAXTOKENSPLIT \
--kptw_lr $KPTWLR
--dataset  $DATASET
--filter $FILTER