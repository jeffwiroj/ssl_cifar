from omegaconf import OmegaConf

def test_base_simsiam_pretrain_config():
    """
    Tests that the base SimSiam pre-training configuration is loaded correctly.
    """
    # Load the configuration file
    conf = OmegaConf.load('configs/base_simsiam_pretrain.yaml')

    # --- Test the 'train' section ---
    assert 'train' in conf, "Top-level key 'train' is missing"
    assert conf.train.backbone == "resnet18"
    assert conf.train.ssl_model == "simsiam"
    assert conf.train.epochs == 800
    assert conf.train.lr == 0.06
    assert conf.train.wd == 5e-4
    assert conf.train.momentum == 0.9
    assert conf.train.lr_warmup == 0
    assert conf.train.use_mixed_precision is True
    assert conf.train.batch_size == 512
    assert conf.train.eval_batch_size == 2048

    # --- Test the 'experiment' section ---
    assert 'experiment' in conf, "Top-level key 'experiment' is missing"
    assert conf.experiment.project_name == "cifar_ssl"
    assert conf.experiment.weight_path == "checkpoints/"
    assert conf.experiment.data_path == "data/"
    assert conf.experiment.resume_from_checkpoint is None
    assert conf.experiment.use_wandb is False
    assert conf.experiment.eval_frequency == 10
    assert conf.experiment.save_frequency == 10
    assert conf.experiment.verbose is False