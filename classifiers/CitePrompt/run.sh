PYTHONPATH=python
SEED=144 # 145 146 147 148
VERBALIZER=kpt #soft auto
FILTER=tfidf_filter # none
KPTWLR=0.06 # 0.06
MAXTOKENSPLIT=-1 # 1
RESULTPATH="results_normal"
DATASET="acl_new"
TESTDATASET="act2_new" # "acl_arc" "scicite" "act2"
TARGET="citation_function" # "citation_function" "citation_object"

$PYTHONPATH citeprompt.py \
--result_file $RESULTPATH \
--seed $SEED \
--verbalizer $VERBALIZER \
--max_token_split $MAXTOKENSPLIT \
--kptw_lr $KPTWLR \
--filter $FILTER \
--dataset $DATASET \
--test_dataset $TESTDATASET \
--target $TARGET