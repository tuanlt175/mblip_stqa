CUDA_VISIBLE_DEVICES=0 python icvrc/evaluate.py \
    --model <your model path> \
    --data_file <your json data file> \
    --images_folder <your images folder> \
    --submission_file public_results.json \
    --question_max_len 48 \
    --answer_max_len 64 \
    --num_beams 2


