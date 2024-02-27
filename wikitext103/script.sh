
##########SOFTMAX BASELINE
 python ./src/train.py --cuda --data data/wikitext-103/ \
--dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 \
--dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 \
--max_step 500000 --attn_type 2 \
--tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 \
--multi_gpu --seed 1111 --job_name softmax \
--work_dir path/to/dir

########## NEUTRENO model
 python ./src/train.py --cuda --data data/wikitext-103/ \
--dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 \
--dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 \
--max_step 500000 --attn_type 222 \
--tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 \
--multi_gpu --seed 1111 --job_name neutreno \
--work_dir path/to/dir



