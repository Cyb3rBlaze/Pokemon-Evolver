class Config():
    def __init__(self):
        self.data_dir = "./cleaned_data"

        self.epochs = 200

        self.batch_size = 8
        
        self.lr = 0.002
        self.beta = 0.5

        self.img_dims = (128, 128)

        self.encoder_in_channels = 64
        self.decoder_in_channels = 256

        self.discriminator_in_channels = 64