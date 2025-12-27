import flexp.runs.make_fedlt_dataset


def main():
    flexp.runs.make_fedlt_dataset.run(
        long_tailed_json_path='data/splitting/cifar10_lt,global,if=50.json',
        client_num=10,
        dirichlet_alpha=0.05,
        output_folder='data/splitting'
    )


if __name__ == '__main__':
    main()
