from base.options.task_options import TaskOptions


class IHCOptions(TaskOptions):

    def __init__(self):
        super().__init__()
        self.parser.add_argument('--num_display_images', type=int, default=3, help="Number of input images to be shown with labels in visualizer")
        self.parser.set_defaults(num_class=2, dataset_name='ihc_patch', )
        pass
