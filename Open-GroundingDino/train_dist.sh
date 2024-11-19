CFG=$1
DATASETS=$2
OUTPUT_DIR=$3

# Set the environment variable for CUDA
export CUDA_VISIBLE_DEVICES=0

python main.py \
    --config_file ${CFG} \
    --datasets ${DATASETS} \
    --output_dir ${OUTPUT_DIR} \
    --pretrain_model_path /home/km/content/groundingdino_swint_ogc.pth \
    --options text_encoder_type="/home/km/content/bert"
