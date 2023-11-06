from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import math
import torch
from tqdm import tqdm

class perplexity_test():
    def __init__(self, model_id):
        super().__init__()
        self.model_id = model_id
        self.model = GPT2LMHeadModel.from_pretrained(self.model_id).to("cuda")
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.model_id)

    def chunks(self, lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def score(self, sentence):
        nlls = []
        self.model.eval()
        # sentence_chunks = list(chunks(sentence, 50))
        with open("ppl_output.txt", "a") as f:
            for sentence_chunk in tqdm(sentence):
                if sentence_chunk!="":
                    tokenize_input = self.tokenizer.encode_plus(str(sentence_chunk), max_length = 200, truncation=True, return_tensors="pt")
                    # try:
                    loss = self.model(tokenize_input.input_ids.to("cuda:1"), labels=tokenize_input.input_ids)["loss"]
                    print(loss.item(), file=f)
                    nlls.append(loss.detach().to("cpu"))
                    # except:
                    #     print(sentence_chunk)
        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl, nlls