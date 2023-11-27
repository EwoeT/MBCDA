


# Requirements
1. torch 2.0.1
2. transformers-4.35.0

# Dataset
csv file with original text and dictionary-based counterfactual text

# Train model
!python mbcda.py -m "facebook/bart-large-cnn" -d "filtered_prepped_wiki_6.csv" -t "text" -l "flipped_text" -s 0.9 -c "cuda" -mode "train"

-t input text column name

-l dictionary-based counterfactual column name -- dictionary-based counterfactuals from eg. [Maudslay et al. 2019](https://github.com/rowanhm/counterfactual-data-substitution/tree/master)

# Generate_text
!python mbcda.py -m "facebook/bart-large-cnn" -d "filtered_prepped_wiki_6.csv" -t "text" -c "cuda" -mode "generate"

-t input text column name

# Pretrained model
[Pretrained MBCDA model](https://drive.google.com/drive/folders/1nG7Hr0GJCa-NDEIMtpv3DpffUuCbK3p0?usp=drive_link)

from transformers import BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained("MBCDA_pretrained_model_path")

CF_ids = model.generate(input_ids, num_beams=3, do_sample=False, min_length=0, max_length=300)
