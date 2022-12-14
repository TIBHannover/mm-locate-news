## MM-Locate-News: Multimodal Focus Location Estimation in News

## Installation

``` bash
# clone the repository
git clone https://github.com/golsa-tahmasebzadeh/mm-locate-news.git
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Download Data
Download the data from here and put in the root directory.

## Download Checkpoints
For mm-locate-news dataset download the trained models from [here](https://tib.eu/cloud/s/j6zmqBtgHm5rs7e) and extract in ``` experiments/snapshots```.
For BreakingNews dataset download the trained models from [here](https://tib.eu/cloud/s/DFPji6E2SKZ3nBf) and extract in ``` breakingnews/experiments/snapshots```.


## Evaluation
To evaluate the models based on mm-locate-news data: 
```bash
python evaluate.py --model_name <MODELNAME> --test_check_point <CHECKPOINT>
```
To evaluate the models based on Breakingnews data: 
```bash
python breakingnews/bn_evaluate.py --model_name <MODELNAME> --test_check_point <CHECKPOINT>
```
To evaluate Cliff-clavin: 
```bash
python Cliff-clavin/evaluate_cliff.py
```

To evaluate Mordecai: 
```bash
python Mordecai/evaluate_mordecai.py
```
To evaluate ISN: 
```bash
python ISN/evaluate_ISN.py
```

## Training 
To train the models based on mm-locate-news dataset:
```bash
python train.py \
--model_name <MODELNAME> \
--resume <CHECKPOINT> \
--freeze_image <[True, False]> \
--freeze_text <[True, False]>
```

To train the models based on BreakingNews dataset:
```bash
python breakingnews/bn_train.py \
--model_name <MODELNAME> \
--resume <CHECKPOINT> 
```

