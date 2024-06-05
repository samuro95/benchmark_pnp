from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from pathlib import Path
    from torch.utils.data import DataLoader
    import deepinv as dinv
    from torchvision import transforms
    from torch.utils.data import DataLoader

# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "deepinv_dataset"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'dataset_name': ["set3c"],
        'img_size': [64],
        'batch_size': [3],
        'num_workers': [0],
        'shuffle': [False],
    }

    def __init__(self, dataset_name='set3c', img_size=256, batch_size=1, num_workers=0, shuffle=False):
        # Store the parameters of the dataset
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.
        # self.dataset_path = None

        BASE_DIR = Path(".")
        DATA_DIR = BASE_DIR / "datasets"

        val_transform = transforms.Compose(
           [transforms.CenterCrop(self.img_size), transforms.ToTensor()]
        )
        dataset = dinv.utils.demo.load_dataset(self.dataset_name, DATA_DIR, transform=val_transform)

        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle
        )

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(dataloader=dataloader)
