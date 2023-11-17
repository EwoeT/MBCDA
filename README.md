


# Requirements
1. torch 2.0.1
2. transformers-4.35.0
3. download the [normalization model](https://drive.google.com/file/d/1XTs9g-BH1Oid8naD8zv1B0qFbgj5pjbV/view?usp=drive_link): For case, punctuation and space correction

# Train model
!python mbcda.py -m "facebook/bart-large-cnn" -p "distilgpt2" -d "filtered_prepped_wiki_6.csv" -t "text" -l "flipped_text" -s 0.9 -c "cuda" -mode "train"

# Generate_text
!python mbcda.py -m "facebook/bart-large-cnn" -p "distilgpt2" -d "filtered_prepped_wiki_6.csv" -t "text" -l "flipped_text" -s 0.9 -c "cuda" -mode "generate"

# Pretrained model
soon to be uploaded
