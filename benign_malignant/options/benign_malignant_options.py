from base.options.task_options import TaskOptions


class BenignMalignantOptions(TaskOptions):

    def __init__(self):
        super().__init__()
        self.parser.add_argument('--num_display_images', type=int, default=3, help="Number of input images to be shown with labels in visualizer")
        self.parser.add_argument('--num_class', type=int, default=2)
        self.parser.add_argument('--model', type=str, default='inception')
        self.parser.add_argument('--dataset_name', type=str, default='gland')
        pass
