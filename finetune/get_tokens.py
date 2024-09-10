import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)


dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"

dna="TAGATGTCCTTGATTAACACCAAAATTAAACCTTTTAAAAACCAGGCATTCAAAAACGGCGAATTCATCGAAATCACCGAA"
dna="AAAGAAAATAATTAATTTTACAGCTGTTAAACCAAACGGTTATAACCTGGTCATACGCAGTAGTTCGGACAAGCGGTACAT"
tokens = tokenizer(dna, return_tensors = 'pt')
ids = tokens["input_ids"]
print(tokens)

for i in ids:
  print("Token::k-mer map:", i, "\t::", tokenizer.decode(i))
