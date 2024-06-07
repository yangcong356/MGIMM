export CUDA_VISIBLE_DEVICES=1

python -m detail_eval_phi \
    --model-path /data1/users/yangcong/code/MultimodalforLongCaption/BMGPG/checkpoints/BMGPG-phi2-2.3b-lora-r16-b1-e5 \
    --model-base microsoft/phi-2 \
    --question-file /data1/users/yangcong/data/GPG/annotations/dior-gpg/dior-gpg-question.json \
    --image-folder /data1/users/yangcong/data/GPG/images \
    --answers-file /data1/users/yangcong/code/MultimodalforLongCaption/BMGPG/scripts/eval/BMGPG-phi2-2.3b-lora-r16-b1-e5-dior-gpg-answer.json \
    --conv-mode llava_phi_2