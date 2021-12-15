python -m util_scripts.generate_video_jpgs ./data/video_data ./data/image_data custom --n_jobs 4
python -m util_scripts.generate_video_split_custom data/video_data data/annotation/ 0.2
python -m util_scripts.custom_json data/annotation data/image_data data/json_file