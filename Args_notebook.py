class args:
#Arguments passed through the code
    # encoders
    encoder_path= None
    encoder_legacy_path= None
    encoder_0_legacy_path= None
    encoder_0_path= None
    encoder_1_legacy_path= None
    encoder_1_path= None
    encoder_lr = 1e-4
    encoder_bs = 128
    encoder_max_epochs = 50
    encoder_type = 'vae'

    # clusters
    cluster_0_path= None
    cluster_1_path= None
    cluster_number = 10
    cluster_path= None
    cluster_n_init = 20
    cluster_max_step = 5000

    # dfc
    dfc_0_path= None
    dfc_1_path= None
    dfc_path= None
    dfc_hidden_dim = 64
    adv_multiplier = 10.0
    dfc_tradeoff = 'none'

    # dec
    dec_lr = 0.001
    dec_batch_size = 512
    dec_iters = 20000

    # dataset
    dataset = "mnist_usps"
    input_height = 32
    digital_dataset = True

    method = "dfc"

    iters = 20000
    lr = 1e-2
    test_interval = 5000
    bs = 512
    log_dir = "./DFC_LOGS/"
    gpu = 0
    seed = 2019

    half_tensor= False

