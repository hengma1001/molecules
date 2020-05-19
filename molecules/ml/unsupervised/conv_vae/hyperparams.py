from molecules.ml.hyperparams import Hyperparams


# TODO: remove num_conv_layers, num_affine_layers since this data is encoded
#       in the lists filters,affine_widths,etc. This will effect keras encoder decoder impl.

# TODO: change member variable names to be the same as pytorch argument names

class ConvVAEHyperparams(Hyperparams):
    def __init__(self, num_conv_layers=3, filters=[64, 64, 64], kernels=[3, 3, 3],
                 strides=[1, 2, 1], latent_dim=3, activation='ReLU', num_affine_layers=1,
                 affine_widths=[128], affine_dropouts=[0], output_activation='Sigmoid'):

        self.num_conv_layers = num_conv_layers 
        self.filters = filters
        self.kernels = kernels
        self.strides = strides
        self.latent_dim = latent_dim
        self.activation = activation
        self.num_affine_layers = num_affine_layers
        self.affine_widths = affine_widths
        self.affine_dropouts = affine_dropouts
        self.output_activation = output_activation

        # Placed after member vars are declared so that base class can validate
        super().__init__()

    def validate(self):
        if len(self.filters) != self.num_conv_layers:
            raise Exception('Number of filters must equal number of convolutional layers.')
        if len(self.kernels) != self.num_conv_layers:
            raise Exception('Number of kernels must equal number of convolutional layers.')
        if len(self.strides) != self.num_conv_layers:
            raise Exception('Number of strides must equal number of convolutional layers.')
        if len(self.affine_widths) != self.num_affine_layers:
            raise Exception('Number of affine width parameters must equal the number of affine layers')
        if len(self.affine_dropouts) != self.num_affine_layers:
            raise Exception('Number of dropout parameters must equal the number of affine layers')

        # Common convention: allows for filter center and for even padding
        if any(kernel % 2 == 0 for kernel in self.kernels):
            raise Exception('Only odd valued kernel sizes allowed')

        if any(p < 0 or p > 1 for p in self.affine_dropouts):
            raise Exception('Dropout probabilities, p, must be 0 <= p <= 1.')
