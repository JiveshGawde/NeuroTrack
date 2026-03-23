import os
import kagglehub

if __name__ == "__main__":

    os.makedirs(os.path.abspath("src/data/alzheimers-data"), exist_ok=True)
    path = kagglehub.dataset_download("yiweilu2033/well-documented-alzheimers-dataset", output_dir=os.path.abspath("src/data/alzheimers-data"))

