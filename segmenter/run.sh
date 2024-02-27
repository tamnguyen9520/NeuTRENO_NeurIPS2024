export DATASET=path/to/data



##### BASELINE
python -m segm.train --log-dir seg_tiny_mask/baseline --dataset ade20k \
--backbone deit_tiny_patch16_224 --decoder mask_transformer --model_name softmax-baseline --attn_type 'softmax' 

###### NEUTRENO
python -m segm.train --log-dir seg_tiny_mask/neutreno --dataset ade20k \
--backbone deit_tiny_patch16_224 --decoder mask_transformer --model_name neutreno --attn_type 'neutreno-former' --alpha 0.6

