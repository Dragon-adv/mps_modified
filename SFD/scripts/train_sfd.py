import os

from flexp.runs import (
    train_sfd_learning_stage,
    train_sfd_stats_agg_stage,
    train_sfd_safs_stage,
    train_sfd_finetune_stage
)
import global_config


def main():
    # Modify global configuration if needed
    # global_config.NUM_WORKERS = ...

    dataset = 'cifar10_lt'
    split_filepath = 'data/splitting/cifar10_lt,dir_label,client=10,if=50,alpha=0.05.json'
    device = 'cuda:0'

    results = train_sfd_learning_stage.run(
        dataset=dataset,
        split_filepath=split_filepath,
        name='SFD_learn,cifar10_lt',
        device=device,
        a_ce_gamma=0.1,
        beta_pi=1.0,
        lr=0.01,
    )
    learned_ckpt = os.path.join(results['result_folder'], 'latest.pt')

    results = train_sfd_stats_agg_stage.run(
        dataset=dataset,
        split_filepath=split_filepath,
        name='SFD_stats,cifar10_lt',
        device=device,
        learned_ckpt=learned_ckpt,
        save_to_ckpt=True,
    )
    rf_model = results['rf_model']
    global_stats = results['global_stats']
    # stats_ckpt = os.path.join(results['result_folder'], 'stats_stage.pt')

    results = train_sfd_safs_stage.run(
        dataset=dataset,
        name='SFD_safs,cifar10_lt',
        device=device,
        # stats_ckpt=stats_ckpt,
        rf_model=rf_model,
        global_stats=global_stats,
        max_syn_num=2000,
        min_syn_num=600,
        steps=100,
        lr=1.0,
        target_cov_eps = 1e-5,
    )
    syn_dataset: list[dict] = results['class_syn_datasets']
    # safs_ckpt = os.path.join(results['result_folder'], 'class_syn_datasets.pt')

    train_sfd_finetune_stage.run(
        dataset=dataset,
        name='SFD_finetune,cifar10_lt',
        device=device,
        syn_dataset=syn_dataset,
        # safs_ckpt=safs_ckpt,
        learned_ckpt=learned_ckpt,
        epochs=100,
        lr=0.01,
        batch_size=64,
        weight_decay=1e-5,
    )


if __name__ == '__main__':
    main()
