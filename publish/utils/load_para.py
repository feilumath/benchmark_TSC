import ml_collections
import numpy as np


class ParaAll(object):
    def __init__(self, *, Mydata_dict=None, Mysize_dict=None):
        if not (Mysize_dict is None): # if both Mydata_dict and Mysize_dict are provided, only use Mysize_dict
            input_layer_size  = Mysize_dict['input_layer_size']
            output_layer_size = Mysize_dict['output_layer_size']
            num_all_samples   = Mysize_dict['num_all_samples']
        else:
            if not (Mydata_dict is None):
                input_layer_size  = Mydata_dict['features'].size()[1]
                output_layer_size = Mydata_dict['labels'].size()[1]
                num_all_samples   = Mydata_dict['labels'].size()[0]
            else:
                raise RuntimeError("We need at least one of Mydata_dict and Mysize_dict as input")
        self.config            = ml_collections.ConfigDict()

        self.config.model      = self.model = ml_collections.ConfigDict()
        self.config.training   = self.training = ml_collections.ConfigDict()
        self.config.opt        = self.opt = ml_collections.ConfigDict()
        self.config.loss       = self.loss = ml_collections.ConfigDict()

        # model
        self.model.input_layer_size   = input_layer_size
        self.model.output_layer_size  = output_layer_size

        # training
        self.training.batch_size   = 16
        self.training.max_epochs   = 1000
        self.training.num_all_samples            = num_all_samples
        self.training.num_tra_samples            = int(np.floor(self.training.num_all_samples * 3 / 4))
        self.training.num_tra_samples_withval    = int(np.floor(self.training.num_all_samples * 2 / 4))
        self.training.num_val_samples_withval    = int(np.floor(self.training.num_all_samples * 1 / 4))

        # optimization
        self.opt.min_lr     = 1e-4
        self.opt.factor     = 0.5
        self.opt.patience   = 50

class ParaResnet(ParaAll):
    def __init__(self, *, Mydata_dict=None, Mysize_dict=None):
        super().__init__(Mydata_dict=Mydata_dict, Mysize_dict=Mysize_dict)

        self.training.max_epochs   = 30

        self.opt.min_lr            = 1e-4
        self.opt.patience          = 5


class ParaMLP(ParaAll):
    def __init__(self, *, Mydata_dict=None, Mysize_dict=None):
        super().__init__(Mydata_dict=Mydata_dict, Mysize_dict=Mysize_dict)

        self.training.max_epochs   = 300

        self.opt.min_lr            = 1e-1
        self.opt.patience          = 50








