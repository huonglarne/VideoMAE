from ipywidgets import Video
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification
from decord import VideoReader, cpu
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-large-finetuned-kinetics")
model.to(device)

video_path = "eating_spaghetti.mp4" 
Video.from_file(video_path, width=500)

feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

# video clip consists of 300 frames (10 seconds at 30 FPS)
vr = VideoReader(video_path, num_threads=1, ctx=cpu(0)) 

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
  converted_len = int(clip_len * frame_sample_rate)
  end_idx = np.random.randint(converted_len, seg_len)
  str_idx = end_idx - converted_len
  index = np.linspace(str_idx, end_idx, num=clip_len)
  index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
  
  return index

vr.seek(0)
index = sample_frame_indices(clip_len=16, frame_sample_rate=4, seg_len=len(vr))
buffer = vr.get_batch(index).asnumpy()
buffer.shape

# create a list of NumPy arrays
video = [buffer[i] for i in range(buffer.shape[0])]

encoding = feature_extractor(video, return_tensors="pt")
print(encoding.pixel_values.shape)

pixel_values = encoding.pixel_values.to(device)

# forward pass
with torch.no_grad():
  outputs = model(pixel_values)
  logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()

print("Predicted class:", model.config.id2label[predicted_class_idx])