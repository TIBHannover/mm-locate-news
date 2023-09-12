# MM-Locate-News: Multimodal Focus Location Estimation in News
This is the official GitHub page for the paper ([link here](https://link.springer.com/chapter/10.1007/978-3-031-27077-2_16)):


## Installation

``` bash
# clone the repository
git clone https://github.com/golsa-tahmasebzadeh/mm-locate-news.git
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Download Data
Download the data from [here](https://tib.eu/cloud/s/cLwMtGoD9QJrRss) and put in the root directory.

## Download Checkpoints
For mm-locate-news dataset download the trained models from [here](https://tib.eu/cloud/s/t3gpgTQg5ZwnabC) and extract in ``` experiments/snapshots```.
For BreakingNews dataset download the trained models from [here](https://tib.eu/cloud/s/7R8GJgm7xP5oTKH) and extract in ``` breakingnews/experiments/snapshots```.


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

## Inference
To get the output predictions for an input image and text pair download [this](https://tib.eu/cloud/s/DSBigrGJnAxnymB) and put it in ```inference/models```.
```bash
python inference/predict.py --test_image_path <IMAGEPATH> --text_input_path <TEXTPATH>
```
## Citation
```
@inproceedings{DBLP:conf/mmm/TahmasebzadehMHE23,
  author       = {Golsa Tahmasebzadeh and
                  Eric M{\"{u}}ller{-}Budack and
                  Sherzod Hakimov and
                  Ralph Ewerth},
  title        = {MM-Locate-News: Multimodal Focus Location Estimation in News},
  booktitle    = {MultiMedia Modeling - 29th International Conference, {MMM} 2023, Bergen,
                  Norway, January 9-12, 2023, Proceedings, Part {I}},
  series       = {Lecture Notes in Computer Science},
  volume       = {13833},
  pages        = {204--216},
  publisher    = {Springer},
  year         = {2023},
  url          = {https://doi.org/10.1007/978-3-031-27077-2\_16},
  doi          = {10.1007/978-3-031-27077-2\_16}
}
```
