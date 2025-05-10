PYTHONPATH=python
DATASET=acl_arc #acl_arc 
SEED=100 # 145 146 147 148
VERBALIZER=kpt #
CALIBRATION="--calibration" # ""
FILTER=tfidf_filter # none
RESULTPATH="results_zeroshot"


$PYTHONPATH zeroshot.py \
--result_file $RESULTPATH \
--dataset $DATASET \
--seed $SEED \
--verbalizer $VERBALIZER $CALIBRATION \
--filter $FILTER

