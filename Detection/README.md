### To train the model from scratch, run the command
```
python3 train.py
```

### To test the model, run the command
```
python3 get_predictions.py ${MODEL}
```
for example,

```
python3 get_predictions.py train_log/models/best.pth
```

### To calculate the mAP on the test dataset with the results of 'get_predictions.py', run the command
```
python3 utils/get_mAP.py ${PRED} ${GT}
```
for example,
```
python3 utils/get_mAP.py train_log/test/predictions.json ../Dataset/Labels/test_full.json
```

### To visualize the images with detection results, run the command
```
python3 utils/make_figures.py ${THRES1} ${THRES2} ${PRED}
```
for example,
```
python3 utils/make_figures.py 0.5 0.5 train_log/test/predictions.json
```
