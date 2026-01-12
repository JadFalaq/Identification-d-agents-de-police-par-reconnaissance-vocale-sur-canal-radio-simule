import kagglehub

# Download latest version
path = kagglehub.dataset_download("pankajsomkuwar/voice-dataset-catalist")

print("Path to dataset files:", path)