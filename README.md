# Emotion from Eyes

Some approaches to using AI to classify emotions from pictures of eyes. The images used to test the approaches are taken from [here](https://embrace-autism.com/reading-the-mind-in-the-eyes-test/#test). Like the linked test, the task is multiple choice. The possible labels for each image are provided in `data.csv`; the correct label is the first one listed (so be sure to shuffle them if otherwise a model might learn to just pick the first one!).

## CLIP

`clip_compare.py` compares embeddings of the possible labels to the embeddings of the image, using [OpenAI's CLIP model](https://openai.com/index/clip/). `open_clip_compare.py` attempts to improve on this by using the larger [Open CLIP](https://github.com/mlfoundations/open_clip) models.

## Multi-modal LLMs

`asking_claude.py` uses Anthropic's Claude API to determine which of the labels is correct.

## Future options

To improve on these results, perhaps a CLIP model could be fine-tuned on a small dataset of example image/emotion pairs.
