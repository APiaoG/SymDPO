#!/bin/bash

<<com
Example Slurm evaluation script. 
Notes:
- VQAv2 test-dev and test-std annotations are not publicly available. 
  To evaluate on these splits, please follow the VQAv2 instructions and submit to EvalAI.
  This script will evaluate on the val split.
com

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

export PYTHONPATH="$PYTHONPATH:open_flamingo"
torchrun --nnodes=1 --nproc_per_node=8  open_flamingo/eval/evaluate.py \
    --vision_encoder_path ViT-L-14 \
    --vision_encoder_pretrained openai\
    --lm_path path_to_anas-awadalla/mpt-1b-redpajama-200b-dolly \
    --lm_tokenizer_path path_to_anas-awadalla/mpt-1b-redpajama-200b-dolly \
    --cross_attn_every_n_layers 1 \
    --checkpoint_path "path_to_checkpoint.pt" \
    --results_file "results_noinstruct_final_vqav2_improve_10k.json" \
    --precision amp_bf16 \
    --batch_size 32 \
    --eval_coco \
    --eval_vqav2 \
    --eval_flickr30 \
    --eval_ok_vqa \
    --eval_textvqa \
    --coco_train_image_dir_path "path_to_coco/train2014" \
    --coco_val_image_dir_path "path_to_coco/val2014" \
    --coco_karpathy_json_path "path_to_karpathy/dataset_coco.json" \
    --coco_annotations_json_path "path_to_coco/annotations/captions_val2014.json" \
    --vqav2_train_image_dir_path "path_to_VQAv2/train2014" \
    --vqav2_train_annotations_json_path "path_to_VQAv2/v2_mscoco_train2014_annotations.json" \
    --vqav2_train_questions_json_path "path_to_VQAv2/v2_OpenEnded_mscoco_train2014_questions.json" \
    --vqav2_test_image_dir_path "path_to_VQAv2/val2014" \
    --vqav2_test_annotations_json_path "path_to_VQAv2/v2_mscoco_val2014_annotations.json" \
    --vqav2_test_questions_json_path "path_to_VQAv2/v2_OpenEnded_mscoco_val2014_questions.json" \
    --flickr_image_dir_path "path_to_Flickr_30K/flickr30k-images" \
    --flickr_karpathy_json_path "path_to_karpathy/dataset_flickr30k.json" \
    --flickr_annotations_json_path "path_to_Flickr_30K/dataset_flickr30k_coco_style.json" \
    --ok_vqa_train_image_dir_path "path_to_coco/train2014" \
    --ok_vqa_train_annotations_json_path "path_to_OKVQA/mscoco_train2014_annotations.json" \
    --ok_vqa_train_questions_json_path "path_to_OKVQA/OpenEnded_mscoco_train2014_questions.json" \
    --ok_vqa_test_image_dir_path "path_to_coco/val2014" \
    --ok_vqa_test_annotations_json_path "path_to_OKVQA/mscoco_val2014_annotations.json" \
    --ok_vqa_test_questions_json_path "path_to_OKVQA/OpenEnded_mscoco_val2014_questions.json" \
    --textvqa_image_dir_path "path_to_textvqa/train_images/" \
    --textvqa_train_questions_json_path "path_to_train_questions_vqa_format.json" \
    --textvqa_train_annotations_json_path "path_to_textvqa/train_annotations_vqa_format.json" \
    --textvqa_test_questions_json_path "path_to_textvqa/val_questions_vqa_format.json" \
    --textvqa_test_annotations_json_path "path_to_textvqa/val_annotations_vqa_format.json" \