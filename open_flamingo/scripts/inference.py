from PIL import Image
import requests
import torch
from open_flamingo import create_model_and_transforms


model, image_processor, tokenizer = create_model_and_transforms(
    "ViT-L-14",
    "openai",
    "path to anas-awadalla/mpt-1b-redpajama-200b-dolly",
    "path to anas-awadalla/mpt-1b-redpajama-200b-dolly",
    cross_attn_every_n_layers=1,
    use_local_files=False,
    gradient_checkpointing=False,
    freeze_lm_embeddings=False,
)


checkpoint = torch.load("path to checkpoint.pt", map_location="cpu")

msd = checkpoint
msd = {k.replace("module.", ""): v for k, v in msd.items()}

model.load_state_dict(msd, False)

"""
Step 1: Load images
"""
demo_image_one = Image.open('path to img')

"""
Step 2: Preprocessing images
Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
 batch_size x num_media x num_frames x channels x height x width. 
 In this case batch_size = 1, num_media = 3, num_frames = 1,
 channels = 3, height = 224, width = 224.
"""
vision_x = [image_processor(demo_image_one).unsqueeze(0)]
vision_x = torch.cat(vision_x, dim=0)
vision_x = vision_x.unsqueeze(1).unsqueeze(0)

"""
Step 3: Preprocessing text
Details: In the text we expect an <image> special token to indicate where an image is.
 We also expect an <|endofchunk|> special token to indicate the end of the text 
 portion associated with an image.
"""
tokenizer.padding_side = "left" # For generation padding tokens should be on the left
lang_x = tokenizer(
    ["<image>Question: question Answer:"],
    return_tensors="pt",
)


"""
Step 4: Generate text
"""
generated_text = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=20,
    num_beams=1,
)

print("Generated text: ", tokenizer.decode(generated_text[0]))