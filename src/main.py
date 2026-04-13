# Main

import cv2
import matplotlib.pyplot as plt
from src.datasets.images_dataset import AlzheimersDataset


if __name__ == "__main__":
    data = AlzheimersDataset(4)

    # images, labels = data[1]

    filtered_data = data.filter(
        label="all", distinct_patients=True, scan="MR1_2")

    print(len(filtered_data))
    # for i, im in enumerate(images):
    #     plt.subplot(4, 1, i+1)
    #     plt.imshow(im.squeeze(0))

    # plt.suptitle(f"Label: {labels.item()} ")
    # plt.show()

    print(filtered_data[:10])
    for j, (patient_id, scan, slice, path, label) in enumerate(filtered_data[:5]):
        plt.subplot(5, 1, j+1)
        plt.title(f"Patient ID: {patient_id} | scan: {scan} | slice: {
                  slice} | label: {label}")
        plt.imshow(cv2.imread(path, cv2.IMREAD_GRAYSCALE), cmap='Grays')

    plt.tight_layout()
    plt.suptitle("Filtered Data")
    plt.show()
