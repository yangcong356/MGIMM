export CUDA_VISIBLE_DEVICES=6
    # --model-base mucai/vip-llava-7b \

python -m detail_eval \
    --model-path /data1/users/yangcong/code/MultimodalforLongCaption/BMGPG/checkpoints/BMGPG-phi2-2.3b-lora-r64-b1-e5 \
    --model-base microsoft/phi-2 \
    --question-file /data1/users/yangcong/data/GPG/annotations/dior-gpg/dior-gpg-question.json \
    --image-folder /data1/users/yangcong/data/GPG/images \
    --answers-file /data1/users/yangcong/code/MultimodalforLongCaption/BMGPG/scripts/eval/BMGPG-phi2-2.3b-lora-r64-b1-e5-dior-gpg-answer.json \
    --conv-mode llava_phi_2