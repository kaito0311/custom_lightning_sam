import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm 


new_state_dict = torch.load("./new_state_dict.pth", map_location="cpu")
original_state_dict =torch.load("weights/mobile_sam.pt", map_location="cpu")

for key in tqdm(new_state_dict.keys()):
    print(key)



exit()


weight = torch.load("weights/mobile_sam.pt", map_location='cpu')

print(weight.keys())


# # Load the CLIP model and processor
# model_name = "openai/clip-vit-base-patch16"
# clip_model = CLIPModel.from_pretrained(model_name)
# clip_processor = CLIPProcessor.from_pretrained(model_name)

# # Input sentence
# input_sentence = "occlusion face object"

# # Tokenize the input sentence
# tokenized_input = clip_processor(input_sentence, return_tensors="pt")
# print(tokenized_input)
# # Generate text embeddings
# with torch.no_grad():
#     text_features = clip_model.get_text_features(**tokenized_input)
#     text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize embeddings

# # Print the shape of the text embeddings
# print("Shape of text embeddings:", text_features.shape)
# print("Text embeddings:", text_features)
# import numpy as np 
# np.save("feature_text", text_features.detach().cpu().numpy())