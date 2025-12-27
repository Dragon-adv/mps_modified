import flexp.runs.make_lt_dataset


def main():
    flexp.runs.make_lt_dataset.run(
        dataset_name='cifar10_lt',
        imbalance_factor=50,
        output_folder='data/splitting'
    )


if __name__ == '__main__':
    main()
