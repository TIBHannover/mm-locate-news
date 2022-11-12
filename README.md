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
To ISN: 
```bash
python Mordecai/evaluate_ISN.py
```

## Training

