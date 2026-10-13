"""Run from the project root: python -m examples.load_data_for_b"""
from src.handoff_data import load_split, read_example


def main():
    train = load_split('train')
    validation = load_split('validation')
    print(f'Training images: {len(train)}; validation images: {len(validation)}')
    image, label = read_example(train.iloc[0])
    print('First training sample:', image.shape, image.dtype, 'grade:', label)
    # image is RGB, HWC (224,224,3), uint8 0..255; label is an int in 0..4.
    # Connect these rows to your chosen framework's Dataset/DataLoader.
    # PyTorch generally requires CHW tensors; TensorFlow generally uses HWC.
    # Apply the exact preprocessing/normalization expected by your chosen weights.
    # Do not normalize twice, relabel to binary, or split again at random.
    # Keep random augmentation and rebalancing restricted to train.
    # The test split is reserved; this handoff is still PROVISIONAL.


if __name__ == '__main__':
    main()
