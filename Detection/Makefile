train:
	python3 train.py

test:
	python3 get_predictions.py ${MODEL}

PRED = train_log/test/predictions.json
GT = ../../Data/Labels/test_full.json
mAP:
	python3 utils/get_mAP.py ${PRED} ${GT}

THRES1 = 0.4
THRES2 = 0.6
show:
	python3 utils/make_figures.py ${THRES1} ${THRES2} ${PRED}
