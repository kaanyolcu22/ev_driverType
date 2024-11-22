import kagglehub

# Download latest version
path = kagglehub.dataset_download("valakhorasani/electric-vehicle-charging-patterns")

print("Path to dataset files:", path)