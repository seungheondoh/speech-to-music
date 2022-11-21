from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, BatchSampler
from stm.classification.music_cls.loader.audioset import AUDIOSET_DATASET

def get_dataloader(args, split):
    dataset = get_dataset(
        dataset_type= args.dataset_type,
        split= split
    )
    if split == "TRAIN":
        data_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False,
        )
    elif split == "VALID":
        data_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False,
        )
    elif split == "TEST":
        data_loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False,
        )
    elif split == "ALL":
        data_loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False,
        )
    return data_loader


def get_dataset(
        dataset_type,
        split,
    ):
    if dataset_type == "Audioset":
        dataset = AUDIOSET_DATASET(split)
    return dataset