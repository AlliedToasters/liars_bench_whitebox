from transformers import AutoTokenizer

def safe_load_tokenizer(model_name):

    if 'mistral' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    return tokenizer