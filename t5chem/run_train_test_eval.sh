#this script is a pipeline to train, test, and evaluate catalyst design models.
#make a dirctory to save models and evaluation results
mkdir example

#train a catalyst design model
stdbuf -oL -eL python run_trainer.py --with_weights 0 --from_scratch 0 --data_dir ../data/AHO/100p_catdesign/ --output_dir example/aho_catdesign/ --task_type product --num_epoc 100 --pretrain Salesforce/codet5-small --num_classes 2 > example/log.train.aho.catdesign 2>&1&

#test the catalyst design model (generate designed catalyst)
stdbuf -oL python run_prediction_nostd.py --data_dir ../data/AHO/100p_catdesign/ --model_dir example/aho_catdesign/ --task_type product --prediction example/catdesign.csv --output_logits 1

#prepare input file (reactant + designed catalyst) for CatScore computation
cp example/catdesign.csv ../data/100p_cattest/
python ../data/100p_cattest/extract_test.py

#train a product prediction model
stdbuf -oL -eL python run_trainer.py --with_weights 1 --from_scratch 0 --data_dir ../data/AHO/100p_prodpred/ --output_dir example/aho_prodpred/ --task_type product --num_epoc 100 --pretrain Salesforce/codet5-small --num_classes 2 > example/log.train.aho.prodpred 2>&1&

#test the product prediction model
stdbuf -oL python run_prediction_nostd.py --data_dir ../data/AHO/100p_prodpred/ --model_dir example/aho_prodpred/checkpoint-58900/ --task_type product --prediction example/prodpred.csv --output_logits 1

#compute CatScore using the product prediction model
stdbuf -oL python run_prediction_nostd.py --data_dir ../data/AHO/100p_cattest/ --model_dir example/aho_prodpred/checkpoint-58900/ --task_type product --prediction example/cattest.csv --output_logits 1

#compute DFTScore for the designed catalyst
cp example/catdesign.csv ../dft/
#in the t5chem/dft/ directory
cd ../dft/
python fill_dft_values.py catdesign.csv
python cal_dftscore_train.py catdesign_filled.csv

#compute Spearman correlation: DFTScore-CatScore, DFTScore-round-trip accuracy
cp example/cattest.csv ../dft/
#in the /t5chem/dft/ director
cd ../dft/
python cal_instance_correlation.py catdesign_filled_dft.csv cattest.csv 

