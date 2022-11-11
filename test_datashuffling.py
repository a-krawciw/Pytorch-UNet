from matplotlib import pyplot as plt

from utils.data_loading import ShuffledDataset

if __name__ == '__main__':
    loader = ShuffledDataset("data/imgs", "data/masks", 720)
    loader[0]
    #plt.imshow(loader[0])
    plt.show()