


#install torch
#install transformers

# Train model
!python mbcda.py -m "facebook/bart-large-cnn" -p "distilgpt2" -d "filtered_prepped_wiki_6.csv" -t "text" -l "flipped_text" -s 0.9 -c "cuda" -mode "train"

# generate_text
!python mbcda.py -m "facebook/bart-large-cnn" -p "distilgpt2" -d "filtered_prepped_wiki_6.csv" -t "text" -l "flipped_text" -s 0.9 -c "cuda" -mode "generate"
