#################################### Inter-skeleton ####################################

# MOCO
CUDA_VISIBLE_DEVICES=0,1 python main_inter_skeleton.py --model moco \
  --lr 0.01 \
  --batch-size 256 \
  --mlp --moco-k 16384 --checkpoint-path ./checkpoints/ntu_60_cross_view/interskeleton_moco_seq_based_graph_based \
  --schedule 351  --epochs 451 --pre-dataset ntu60 \
  --skeleton-representation seq-based_and_graph-based --protocol cross_view

# Siamese
CUDA_VISIBLE_DEVICES=0,1 python main_inter_skeleton.py --model simsiam \
  --lr 0.01 \
  --batch-size 256 \
  --mlp --checkpoint-path ./checkpoints/ntu_60_cross_view/interskeleton_byol_seq_based_graph_based \
  --schedule 351 --epochs 451 --pre-dataset ntu60 \
  --skeleton-representation seq-based_and_graph-based --protocol cross_view


#################################### Intra-skeleton ####################################

CUDA_VISIBLE_DEVICES=0,1 python main_intra_skeleton.py \
--model simsiam --lr 0.1 --batch-size 256 --mlp \
--checkpoint-path ./checkpoints/prova_2/ --schedule 200 300 --epochs 400 \
--pre-dataset ntu60 --skeleton-representation graph-based --protocol cross_view