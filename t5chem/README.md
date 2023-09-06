## We can reduce the model size to produce a range of models
To train a reduced size model, we can specify the model directory

```
# one-five layers are removed from the "small" version of Code-T5
python remove_layers.py --pretrain Salesforce/codet5-small

# from the five-layer-removed model after running the above script, we can further slice down the model
python slice_model.py --pretrain pruned/Salesforce/codet5-small/5_removed

# for example, to train a three-layer removed model for catayst design, run the following script
stdbuf -oL -eL python run_trainer.py --with_weights 0 --from_scratch 0 --data_dir path/to/data/AHO/100p_catdesign/ --output_dir example/aho_catdesign_3removed/ --task_type product --num_epoc 100 --pretrain ${WORKING_DIR}$/pruned/Salesforce/codet5-small/3_removed --num_classes 2 > example/log.train.aho.catdesign.3removed.loss 2>&1&
```

