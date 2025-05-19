from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Specify the dataset
dataset = "sbhatti/financial-sentiment-analysis"

# Download and unzip into ./data
api.dataset_download_files(
    dataset,
    path="data",        # where to put the files
    unzip=True,         # unzip after download
    quiet=False         # show progress
)

print("Dataset downloaded to ./data")
