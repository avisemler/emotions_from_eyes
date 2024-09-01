import base64
import csv

import anthropic
from PIL import Image

SECRET_KEY = "[ADD SECRET KEY HERE]"

client = anthropic.Anthropic(
    api_key=SECRET_KEY,
)

def get_label(image_path, label_options, file_format="img/jpeg"):
    prompt = "Please judge what emotion is shown in this image, out of the following options: "
    prompt += ", ".join(label_options)
    prompt += ". Answer with only single word! If you are unsure, guess."

    with open(image_path, "rb") as f:
        image_text = base64.b64encode(f.read()).decode("utf-8")

    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=16,
        temperature=0.0,
        messages=[
            {"role": "user", "content": [
                    {
                        "type": "image",
                        "source": {
                        "type": "base64",
                        "media_type": file_format,
                        "data": image_text,
                        }
                    },
                    {"type": "text", "text": prompt}
                    ]}
        ]
    )
    return message.content[0].text.lower().replace(".", "")


if __name__ == "__main__":
    count = 0
    correct_count = 0
    with open("data.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                #the correct label is always the first one
                count += 1
                predicted = get_label(row[0], row[1:])
                print("True emotion:", row[1], " - Predicted:", predicted)
                if predicted == row[1]:
                    correct_count += 1

    print("Accuracy (%):", 100 * correct_count/count)