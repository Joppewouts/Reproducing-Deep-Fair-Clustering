class args:
        #Arguments passed through the code
        #1
        def set_mnist_ups(self):
                # encoders
                self.encoder_path= './FINAL_MODELS/MNIST/VAE_mnist.ckpt'
                self.encoder_legacy_path= None
                self.encoder_0_legacy_path= None #unnecesary, used to load original paper models
                self.encoder_0_path= None
                self.encoder_1_legacy_path= None#unnecesary, used to load original paper models
                self.encoder_1_path= None
                self.encoder_lr = 0.001
                self.encoder_bs = 64
                self.encoder_max_epochs = 50
                self.encoder_type = 'vae'

                # clusters
                self.cluster_0_path= None
                self.cluster_1_path= None
                self.cluster_number = 10
                self.cluster_path= './FINAL_MODELS/MNIST/clusters_dfc.txt'
                self.cluster_n_init = 20
                self.cluster_max_step = 5000

                # dfc
                self.dfc_0_path= None
                self.dfc_1_path= None
                self.dfc_path= './FINAL_MODELS/MNIST/DFC_DFC.pth'
                self.dfc_hidden_dim = 64
                self.adv_multiplier = 10.0
                self.dfc_tradeoff = 'none'

                # dec
                self.dec_lr = 0.0001
                self.dec_batch_size = 256
                self.dec_iters = 50000

                # dataset
                self.dataset = "mnist_usps"
                self.input_height = 32
                self.digital_dataset = True

                self.method = "dfc"

                self.iters = 20000
                self.lr = 0.001
                self.test_interval = 100
                self.bs = 64
                self.log_dir = "./jupyter_results_mnist_ups/"
                self.gpu = 0
                self.seed = 2019

                self.half_tensor= False
        #2
        def set_reverse_mnist(self):
               # encoders
                self.encoder_path= './FINAL_MODELS/MNIST/VAE_mnist.ckpt'
                self.encoder_legacy_path= None
                self.encoder_0_legacy_path= None
                self.encoder_0_path= None
                self.encoder_1_legacy_path= None
                self.encoder_1_path= None
                self.encoder_lr = 0.001
                self.encoder_bs = 64
                self.encoder_max_epochs = 50
                self.encoder_type = 'vae'

                # clusters
                self.cluster_0_path= None
                self.cluster_1_path= None
                self.cluster_number = 10
                self.cluster_path= './FINAL_MODELS/MNIST/clusters_dfc.txt'
                self.cluster_n_init = 20
                self.cluster_max_step = 5000

                # dfc
                self.dfc_0_path= None
                self.dfc_1_path= None
                self.dfc_path= './FINAL_MODELS/MNIST/DFC_DFC.pth'
                self.dfc_hidden_dim = 64
                self.adv_multiplier = 10.0
                self.dfc_tradeoff = 'none'

                # dec
                self.dec_lr = 0.0001
                self.dec_batch_size = 256
                self.dec_iters = 50000

                # dataset
                self.dataset = "reverse_mnist"
                self.input_height = 32
                self.digital_dataset = True

                self.method = "dfc"

                self.iters = 20000
                self.lr = 0.001
                self.test_interval = 100
                self.bs = 64
                self.log_dir = "./jupyter_results_reverse_mnist/"
                self.gpu = 0
                self.seed = 2019

                self.half_tensor= False
        #3
        def set_mtfl(self):
                 # encoders
                self.encoder_path=None # './FINAL_MODELS/MTFL/RESNET50_mtfl.pth'
                self.encoder_legacy_path= None
                self.encoder_0_legacy_path= None
                self.encoder_0_path= None#'./FINAL_MODELS/MTFL/RESNET50_mtfl_0.pth'
                self.encoder_1_legacy_path= None
                self.encoder_1_path= None#'./FINAL_MODELS/MTFL/RESNET50_mtfl_1.pth'
                self.encoder_lr = 1e-4
                self.encoder_bs = 128
                self.encoder_max_epochs = 50
                self.encoder_type = 'resnet50'

                # clusters
                self.cluster_0_path= './FINAL_MODELS/MTFL/mtfl_cluster_0.txt'
                self.cluster_1_path= './FINAL_MODELS/MTFL/mtfl_cluster_1.txt'
                self.cluster_number = 2
                self.cluster_path= './FINAL_MODELS/MTFL/mtfl_cluster.txt'#mtfl_cluster_dec.txt
                self.cluster_n_init = 20
                self.cluster_max_step = 5000

                # dfc
                self.dfc_0_path= './FINAL_MODELS/MTFL/MTFL/DEC_mtfl_0.pth'
                self.dfc_1_path= './FINAL_MODELS/MTFL/DEC_mtfl_1.pth'
                self.dfc_path= './FINAL_MODELS/MTFL/DFC_DFC_mtfl.pth' #DFC_DEC_mtfl.pth DEC method
                self.dfc_hidden_dim = 1000
                self.adv_multiplier = 10.0
                self.dfc_tradeoff = 'none'

                # dec
                self.dec_lr = 0.0001 
                self.dec_batch_size = 128
                self.dec_iters = 20000 

                # dataset
                self.dataset = "mtfl"
                self.input_height = 32
                self.digital_dataset = True

                self.method = "dfc"

                self.iters = 20000
                self.lr = 0.001
                self.test_interval = 100
                self.bs = 128
                self.log_dir = "./jupyter_results_mtfl/"
                self.gpu = 0
                self.seed = 2019

                self.half_tensor= False
        def set_mtfl_load_models(self):
                 # encoders
                self.encoder_path= './FINAL_MODELS/MTFL/RESNET50_mtfl_encoder.pth'
                self.encoder_legacy_path= None
                self.encoder_0_legacy_path= None
                self.encoder_0_path= None#'./FINAL_MODELS/MTFL/RESNET50_mtfl_0.pth'
                self.encoder_1_legacy_path= None
                self.encoder_1_path= None#'./FINAL_MODELS/MTFL/RESNET50_mtfl_1.pth'
                self.encoder_lr = 1e-4
                self.encoder_bs = 128
                self.encoder_max_epochs = 50
                self.encoder_type = 'resnet50'

                # clusters
                self.cluster_0_path= './FINAL_MODELS/MTFL/mtfl_cluster_0.txt'
                self.cluster_1_path= './FINAL_MODELS/MTFL/mtfl_cluster_1.txt'
                self.cluster_number = 2
                self.cluster_path= './FINAL_MODELS/MTFL/mtfl_cluster.txt'#mtfl_cluster_dec.txt
                self.cluster_n_init = 20
                self.cluster_max_step = 5000

                # dfc
                self.dfc_0_path= './FINAL_MODELS/MTFL/MTFL/DEC_mtfl_0.pth'
                self.dfc_1_path= './FINAL_MODELS/MTFL/DEC_mtfl_1.pth'
                self.dfc_path= './FINAL_MODELS/MTFL/DFC_DFC_mtfl.pth' #DFC_DEC_mtfl.pth DEC method
                self.dfc_hidden_dim = 1000
                self.adv_multiplier = 10.0
                self.dfc_tradeoff = 'none'

                # dec
                self.dec_lr = 0.0001 
                self.dec_batch_size = 128
                self.dec_iters = 20000 

                # dataset
                self.dataset = "mtfl"
                self.input_height = 32
                self.digital_dataset = True

                self.method = "dfc"

                self.iters = 20000
                self.lr = 0.001
                self.test_interval = 100
                self.bs = 128
                self.log_dir = "./jupyter_results_mtfl/"
                self.gpu = 0
                self.seed = 2019

                self.half_tensor= False
        #4
        def set_office31(self):
                 # encoders
                self.encoder_path= None#'./FINAL_MODELS/Office31/Office31_DFC_DFC_encoder'
                self.encoder_legacy_path= None
                self.encoder_0_legacy_path= None
                self.encoder_0_path= None
                self.encoder_1_legacy_path= None
                self.encoder_1_path= None
                self.encoder_lr = 1e-4
                self.encoder_bs = 128
                self.encoder_max_epochs = 50
                self.encoder_type = 'resnet50'

                # clusters
                self.cluster_0_path= None
                self.cluster_1_path= None
                self.cluster_number = 31
                self.cluster_path= None
                self.cluster_n_init = 20
                self.cluster_max_step = 5000

                # dfc
                self.dfc_0_path= None
                self.dfc_1_path= None
                self.dfc_path= None
                self.dfc_hidden_dim = 1000
                self.adv_multiplier = 10.0
                self.dfc_tradeoff = 'none'

                # dec
                self.dec_lr = 0.0001
                self.dec_batch_size = 128
                self.dec_iters = 20000

                # dataset
                self.dataset = "office_31"
                self.input_height = 32
                self.digital_dataset = True

                self.method = "dfc"

                self.iters = 20000
                self.lr = 0.001
                self.test_interval = 100
                self.bs = 128
                self.log_dir = "./jupyter_results_office31/"
                self.gpu = 0
                self.seed = 2019

                self.half_tensor= False

        def set_office31_load_models(self):
                 # encoders
                self.encoder_path= './FINAL_MODELS/Office31/Office31_DFC_DFC_encoder.pth'
                self.encoder_legacy_path= None
                self.encoder_0_legacy_path= None
                self.encoder_0_path= None
                self.encoder_1_legacy_path= None
                self.encoder_1_path= None
                self.encoder_lr = 1e-4
                self.encoder_bs = 128
                self.encoder_max_epochs = 50
                self.encoder_type = 'resnet50'

                # clusters
                self.cluster_0_path= None
                self.cluster_1_path= None
                self.cluster_number = 31
                self.cluster_path= 'clusters_dfc.txt'
                self.cluster_n_init = 20
                self.cluster_max_step = 5000

                # dfc
                self.dfc_0_path= None
                self.dfc_1_path= None
                self.dfc_path= './FINAL_MODELS/Office31/DFC_DFC.pth'
                self.dfc_hidden_dim = 1000
                self.adv_multiplier = 10.0
                self.dfc_tradeoff = 'none'

                # dec
                self.dec_lr = 0.0001
                self.dec_batch_size = 128
                self.dec_iters = 20000

                # dataset
                self.dataset = "office_31"
                self.input_height = 32
                self.digital_dataset = True

                self.method = "dfc"

                self.iters = 20000
                self.lr = 0.001
                self.test_interval = 100
                self.bs = 128
                self.log_dir = "./jupyter_results_office31/"
                self.gpu = 0
                self.seed = 2019

                self.half_tensor= False

        