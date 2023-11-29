


# Requirements
1. torch 2.0.1
2. transformers-4.35.0

# Dataset
csv file with original text and dictionary-based counterfactual text

# Train model
```python
!python mbcda.py -m "facebook/bart-large-cnn" -d "filtered_prepped_wiki_6.csv" -t "text" -l "flipped_text" -s 0.9 -c "cuda" -mode "train"

-t input text column name

-l dictionary-based counterfactual column name -- dictionary-based counterfactuals from eg. [Maudslay et al. 2019](https://github.com/rowanhm/counterfactual-data-substitution/tree/master)
```
<!---
# Generate_text
```python
!python generate_mbcda.py -m "facebook/bart-base" -data_path "../data/bias_in_bios_original" -model_path "pretrained_MBCDA_epoch_1" -mode "generate" -device "cuda"
```
-->
# Pretrained model

download [Pretrained MBCDA model](https://drive.google.com/drive/folders/1nG7Hr0GJCa-NDEIMtpv3DpffUuCbK3p0?usp=drive_link)

## Generate_text

### txt file input
```python
!python generate_mbcda.py -m "facebook/bart-base" -data_path "path_to_data" -model_path "pretrained_MBCDA_epoch_1" -num_beams 10 -mode "generate" -device "cuda"```
```

### sentence list input
```python
device = "cuda"
from transformers import AutoTokenizer, BartForConditionalGeneration
model = BartForConditionalGeneration.from_pretrained("MBCDA_pretrained_model").to(device).eval()
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
inputs = tokenizer.batch_encode_plus(input_batchh, truncation=True, max_length=300, return_tensors="pt", padding=True).to(device)
input_ids = inputs["input_ids"]
summary_ids = model.generate(input_ids, num_beams=8, do_sample=False, min_length=0, max_length=200)
output_text = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
```
