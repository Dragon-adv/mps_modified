import flexp.runs.test
import global_config


def main():
    # Modify global configuration if needed
    # global_config.NUM_WORKERS = ...

    dataset = 'cifar10_lt'
    model_ckpt = 'output/results/xxx/xxx.pt'
    device = 'cuda:0'
    flexp.runs.test.run(
        dataset=dataset,
        model_ckpt=model_ckpt,
        device=device,
    )


if __name__ == '__main__':
    main()
