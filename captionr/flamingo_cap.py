from PIL import Image
import torch
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import os

SUPPORTED_EXT = ['.jpg', '.png']  # Add more extensions if needed

class Flamingo:
    model_name = "openflamingo/OpenFlamingo-9B-vitl-mpt7b"
    device = None
    dtype = None
    model = None
    image_processor = None
    tokenizer = None

    def __init__(self, device, model_name=None, force_cpu=False, example_root=None, **kwargs) -> None:
        if model_name is not None:
            self.model_name = model_name
        self.device = device
        self.dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-7b",
            tokenizer_path="anas-awadalla/mpt-7b",
            cross_attn_every_n_layers=1,
        )
        self.tokenizer.padding_side = "left"
        checkpoint_path = hf_hub_download(self.model_name, "checkpoint.pt")
        self.model.load_state_dict(torch.load(checkpoint_path), strict=False)
        self.model.to(0, dtype=self.dtype)
        self.examples = self.load_examples(example_root)

    def load_examples(self, example_root):
        examples = []
        if example_root is not None:
            for root, dirs, files in os.walk(example_root):
                for file in files:
                    ext = os.path.splitext(file)[-1].lower()
                    if ext in SUPPORTED_EXT:
                        txt_file = os.path.splitext(file)[0] + ".txt"
                        with open(os.path.join(root, txt_file), 'r') as f:
                            caption = f.read()
                        image = Image.open(os.path.join(root, file))
                        vision_x = [self.image_processor(image).unsqueeze(0)]
                        examples.append((caption, vision_x))
        return examples

    def caption(self, img: Image, **kwargs) -> str:
        # Add the new image to the examples
        vision_x = [vx[1][0] for vx in self.examples]
        vision_x.append(self.image_processor(img).unsqueeze(0))
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)
        vision_x = vision_x.to(self.device, dtype=self.dtype)

        # Generate the prompt
        prompt = ""
        output_prompt = "Output:"
        per_image_prompt = "<image> " + output_prompt
        for example in iter(self.examples):
            prompt += f"{per_image_prompt}{example[0]}"
        prompt += per_image_prompt # prepare for novel example
        prompt = prompt.replace("\n", "") # in case captions had newlines

        # Generate the captions
        generated = self.model.generate(
            input_ids=self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device),
            decoder_start_token_id=self.model.config.pad_token_id,
            min_length=len(self.tokenizer(prompt, return_tensors="pt").input_ids[0]) + kwargs.get('min_new_tokens', 20),
            max_length=len(self.tokenizer(prompt, return_tensors="pt").input_ids[0]) + kwargs.get('max_new_tokens', 50),
            num_beams=kwargs.get('num_beams', 3),
            temperature=kwargs.get('temperature', 1.0),
            top_k=kwargs.get('top_k', 0),
            top_p=kwargs.get('top_p', 0.9),
            repetition_penalty=kwargs.get('repetition_penalty', 1.0),
            length_penalty=kwargs.get('length_penalty', 1.0),
            vision_data=vision_x,
        )

        # Decode the generated captions
        print(f"Generated: {generated}")
        decoded = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Decoded: {decoded}")
        caption = decoded.split(output_prompt)[-1].strip()
        print(f"Caption: {caption}")
        return caption
    
    def process_img(self, img_path):
        with Image.open(img_path).convert('RGB') as img:
            return self.caption(img)

    def remove_duplicates(self, string):
        words = string.split(', ')
        unique_words = []
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
            else:
                break
        return ', '.join(unique_words)
