from GraphDataset import MyOwnDataset

def main():
    dataset = MyOwnDataset(root='UMRDataset/')

    print(dataset.num_features, len(dataset))

if __name__ == "__main__":
    main()