import csv

import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_label_probs(image_path, label_options):
    """Get probabilities for which label is the correct one for the image."""
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(label_options).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
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