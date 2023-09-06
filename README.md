# CatScore

This repo provides code for "CatScore: A High-Efficiency Evaluation Metric for Asymmetric Catalyst Design" by Bing Yan and Kyunghyun Cho. This repo is based off [T5Chem](https://github.com/HelloJocelynLu/t5chem) by Jieyu Lu and Yingkai Zhang. Note that this repo can be found at https://github.com/bingyan4science/CatScore.

## Dependencies

The code has been tested on Python 3.8 and PyTorch 1.12.1.

* Python 3.8: `conda create --prefix ../conda_catscore python=3.8`
* PyTorch: https://pytorch.org/get-started/locally/ `conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge`

In addition, install dependencies using

```
pip install -r requirements.txt
```

## Installation

```
python setup.py install
cd t5chem
export WORKING_DIR=$(pwd)
```


## Usage

Note that we always assume the working directory to be `WORKING_DIR`:

```
cd ${WORKING_DIR}
```
## Make a dirctory to save models and evaluation results
```
mkdir example
```
Note that your data are saved under the data directory ../data/AHO/100p_catdesign, ../data/AHO/100p_prodpred, and ../data/AHO/100p_cattest

## Train a catalyst design model and generate from it

This project provides an automatic evaluation metric for catalyst design models, so we first need a catalyst design model. We will train such a model by finetuning a pretrained codet5. I am using the 100 percent setting (see my report for more details) as an example. For other settings simply change `100percent` to other percentages:

```
stdbuf -oL -eL python run_trainer.py --with_weights 0 --from_scratch 0 --data_dir ../data/AHO/100p_catdesign/ --output_dir example/aho_catdesign/ --task_type product --num_epoc 100 --pretrain Salesforce/codet5-small --num_classes 2 > example/log.train.aho.catdesign 2>&1&
```

Next, we generate catalysts using the trained model.
```
stdbuf -oL python run_prediction_nostd.py --data_dir ../data/AHO/100p_catdesign/ --model_dir example/aho_catdesign/ --task_type product --prediction example/catdesign.csv --output_logits 1
```

## Prepare input file (reactant + designed catalyst) for CatScore computation
```
cp example/catdesign.csv ../data/100p_cattest/
# Now change to the data directory and extract the catalyst information to generate test.source file
cd ../data/100p_cattest
python extract_test.py
```

## Train a product prediction model

The core idea in CatScore is to use a product prediction model to replace running real chemistry experiments to test how good the designed catalysts are. Therefore, we need to train a product prediction model.

```
cd ${WORKING_DIR}$
stdbuf -oL -eL python run_trainer.py --with_weights 1 --from_scratch 0 --data_dir ../data/AHO/100p_prodpred/ --output_dir example/aho_prodpred/ --task_type product --num_epoc 100 --pretrain Salesforce/codet5-small --num_classes 2 > example/log.train.aho.prodpred 2>&1&

#test the product prediction model
stdbuf -oL python run_prediction_nostd.py --data_dir ../data/AHO/100p_prodpred/ --model_dir example/aho_prodpred/checkpoint-58900/ --task_type product --prediction example/prodpred.csv --output_logits 1
```

## Compute CatScore

To compute CatScore, we use the predicted catalysts by the catalyst design model as input, and evaluate the likelihood of producing the target product under the product prediction model:

```
stdbuf -oL python run_prediction_nostd.py --data_dir ../data/AHO/100p_cattest/ --model_dir example/aho_prodpred/checkpoint-58900/ --task_type product --prediction example/cattest.csv --output_logits 1
```

Now we can compute CatScore of the system by simply taking the exponential of the total log likelihood. We can also calculate the round-trip accuracy based on the rank of the target product - if the target product rank at 1, the round-trip accuracy is 1; otherwise, 0. The above calcualted CatScore and round-trip accuracy are both at the instance level. The system-level scores are simply an average of the instance-level scores.

```
cd example
python cal_catscore_accuracy.py cattest.csv
```

## Compute DFTScore for the designed catalyst. This is done under the directory ../dft/
```
cd ${WORKING_DIR}$
cp example/catdesign.csv ../dft/
cd ../dft/
python fill_dft_values.py catdesign.csv
python cal_dftscore_train.py catdesign_filled.csv
```

## Now we can calcualte Spearman correlation: DFTScore-CatScore, DFTScore-round-trip accuracy
```
cd ${WORKING_DIR}$
cp example/cattest.csv ../dft/
cd ../dft/
python cal_instance_correlation.py catdesign_filled_dft.csv cattest.csv
```

