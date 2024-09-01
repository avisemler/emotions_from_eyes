import csv

import torch
import open_clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu', pretrained='dfn5b')
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')

def get_label_probs(image_path, label_options):
    """Get probabilities for which label is the correct one for the image."""
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = tokenizer(label_options).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        probs = text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()
    return probs

if __name__ == "__main__":
    count = 0
    correct_count = 0
    with open("data.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                #the correct label is always the first one
                count += 1
                predicted = get_label_probs(row[0], row[1:]).argmax()
                print("True emotion:", row[1], " - Predicted:", row[1:][predicted])
                if predicted == 0:
                    correct_count += 1

    print("Accuracy (%):", 100 * correct_count/count)