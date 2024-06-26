from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def __init__(self):
        super().__init__()
        
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.set_defaults(phase='test', batch_size=1)
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--test_split', type=str, default='test', help='which split to evaluate on')
        parser.add_argument('--test_epoch_vis_n', type=int, default=20, help='number of data to visualize')
        self.is_train, self.is_test, self.is_infer = False, True, False
        return parser
