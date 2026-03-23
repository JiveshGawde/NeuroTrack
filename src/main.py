# Main

import matplotlib.pyplot as plt 
from src.datasets.images_dataset import AlzheimersDataset


if __name__ == "__main__":
    data = AlzheimersDataset(4)

    images, labels = data[2]
    
    for i, im in enumerate(images):
        plt.subplot(4, 1, i+1)
        plt.imshow(im.squeeze(0))

    plt.suptitle(f"Label: {labels.item()} ")
    plt.show()
