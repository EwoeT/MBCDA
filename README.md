


# Requirements
1. torch 2.0.1
2. transformers-4.35.0
3. download the [normalization model](): For case, punctuation and space correction

# Train model
!python mbcda.py -m "facebook/bart-large-cnn" -p "distilgpt2" -d "filtered_prepped_wiki_6.csv" -t "text" -l "flipped_text" -s 0.9 -c "cuda" -mode "train"

# generate_text
!python mbcda.py -m "facebook/bart-large-cnn" -p "distilgpt2" -d "filtered_prepped_wiki_6.csv" -t "text" -l "flipped_text" -s 0.9 -c "cuda" -mode "generate"
