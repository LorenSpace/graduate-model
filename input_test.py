import argparse
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers.trainer_utils import set_seed
from trainer import PET_layer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def detect_euphemism(text, args):
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    special_tokens_dict = {'additional_special_tokens': ['[START_EUPH]', '[END_EUPH]']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model = AutoModel.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))

    # if args.model_type == "cls":
    #     model.cls_layer = CLS_Layer(args.pet_dim, device)
    # elif args.model_type == "pet":
    model.pooler = nn.Identity()
    model.pet = PET_layer.PET_layer(tokenizer, args.pet_dim, device)
    # elif args.model_type == "dan":
    #     model.pooler = nn.Identity()
    #     model.pet = Sent_DAN_Simple(tokenizer, args.pet_dim, device)
    # else:
    #     raise NotImplementedError

    model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))

    text = text.replace("<", "[START_EUPH] ").replace(">", " [END_EUPH]")
    inputs = tokenizer(text, return_tensors='pt', max_length=args.max_length, padding="max_length", truncation=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    if args.model_type == "cls":
        logits = model.cls_layer(outputs['pooler_output'])
    else:
        last_hidden_state = outputs['last_hidden_state']
        logits = model.pet(last_hidden_state, input_ids)

    prediction = torch.argmax(logits, dim=1).item()
    return prediction


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str, default='roberta-large')
#     parser.add_argument("--model_type", type=str, default='pet')
#     parser.add_argument("--model_path", type=str, required=True)
#     parser.add_argument("--max_length", type=int, default=256)
#     parser.add_argument("--seed", type=int, default=111)
#     args = parser.parse_args()
#     args.pet_dim = 1024 if "large" in args.model else 768

    # Example usage:
    # text = "I am <happy> to see you."
    # print(f"Euphemism Detected: {detect_euphemism(text, args)}")
