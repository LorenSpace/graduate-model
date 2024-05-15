import torch
from torch import nn


class PET_layer(nn.Module):
    def __init__(self, tokenizer, pet_dim, device):
        super(PET_layer, self).__init__()
        self.tokenizer = tokenizer
        self.pet_dim = pet_dim
        self.device = device
        self.linear1 = nn.Linear(pet_dim, pet_dim)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(pet_dim, 2)

    def forward(self, inputs, input_ids):
        euph_tensor = torch.zeros([inputs.shape[0], inputs.shape[-1]]).to(self.device)
        for i in range(input_ids.shape[0]):
            idxes = extract_euph_idx(self.tokenizer, input_ids[i])
            for j in idxes:
                euph_tensor[i] += inputs[i][j]
        out = self.linear2(self.dropout(self.linear1(euph_tensor)))
        return out


def extract_euph_idx(tokenizer, input):
    """
    input is list of numbers
    """
    start_euph_idx = len(tokenizer) - 2
    start_idx = (input == start_euph_idx).nonzero().squeeze()
    end_idx = (input == start_euph_idx + 1).nonzero().squeeze()
    euph_idx = [idx for idx in range(start_idx + 1, end_idx)]
    return euph_idx

# class PET_layer(nn.Module):
#     def __init__(self, tokenizer, pet_dim, device):
#         super(PET_layer, self).__init__()
#         self.tokenizer = tokenizer
#         self.pet_dim = pet_dim
#         self.device = device
#         self.linear1 = nn.Linear(pet_dim, pet_dim)
#         self.dropout = nn.Dropout(0.1)
#         self.linear2 = nn.Linear(pet_dim, 2)
#
#     def forward(self, inputs, input_ids):
#         euph_tensor = torch.zeros([inputs.shape[0], self.pet_dim]).to(self.device)
#         for i, input_id in enumerate(input_ids):
#             idxes = extract_euph_idx(self.tokenizer, input_id)
#             if idxes:
#                 euph_tensor[i] = inputs[i, idxes].sum(dim=0)  # Summing all embeddings of euphemism indices
#         out = self.linear2(self.dropout(self.linear1(euph_tensor)))
#         return out

#
# def extract_euph_idx(tokenizer, input):
#     """
#     input is a tensor of token indices
#     """
#     start_euph_idx = len(tokenizer) - 2  # Assuming the last two tokens are special euphemism markers
#     start_idx = (input == start_euph_idx).nonzero(as_tuple=True)[0]
#     end_idx = (input == start_euph_idx + 1).nonzero(as_tuple=True)[0]
#
#     if len(start_idx) == 0 or len(end_idx) == 0:
#         return []  # No euphemism markers found
#
#     # Assuming the first start marker and the first end marker enclose the euphemism
#     if start_idx[0] < end_idx[0]:
#         euph_idx = list(range(start_idx[0].item() + 1, end_idx[0].item()))
#         return euph_idx
#     else:
#         return []  # No valid euphemism interval found
