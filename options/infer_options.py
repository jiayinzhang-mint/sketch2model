from datetime import datetime

from .base_options import BaseOptions


class InferOptions(BaseOptions):
    """This class includes inference options.

    It also includes shared options defined in BaseOptions.
    """

    def __init__(self):
        super().__init__()

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.set_defaults(phase='infer', dataset_mode='inference')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')

        parser.add_argument('--data_name', type=str, default=datetime.now().strftime("%Y%m%d%H%M%S"),
                            help='identifier to distinguish different runs')
        parser.add_argument('--image_path', type=str, default='./load/inference/car.png', required=True,
                            help='path to input image')
        parser.add_argument('--view', type=float, nargs=2, required=False,
                            help='specified view, in the format of [elevation azimuth]')

        self.is_train, self.is_test, self.is_infer = False, False, True
        return parser
