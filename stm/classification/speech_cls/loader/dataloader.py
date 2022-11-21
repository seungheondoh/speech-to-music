from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, BatchSampler
from stm.classification.speech_cls.loader.iemocap import IEMOCAP_Dataset
from stm.classification.speech_cls.loader.hikia import HIKIA_Dataset
from stm.classification.speech_cls.loader.ravdess import RAVDESS_Dataset

def get_dataloader(args, split):
    dataset = get_dataset(
        dataset_type= args.dataset_type,
        split= split
    )
    if split == "TRAIN":
        data_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False,
            collate_fn=dataset.batch_processor
        )
    elif split == "VALID":
        data_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False,
            collate_fn=dataset.batch_processor
        )
    elif split == "TEST":
        data_loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False,
            collate_fn=dataset.batch_processor
        )
    elif split == "ALL":
        data_loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False,
            collate_fn=dataset.batch_processor
        )
    return data_loader


def get_dataset(
        dataset_type,
        split,
    ):
    if dataset_type == "IEMOCAP":
        dataset = IEMOCAP_Dataset(split)
    elif dataset_type == "HIKIA":
        dataset = HIKIA_Dataset(split)
    elif dataset_type == "RAVDESS":
        dataset = RAVDESS_Dataset(split)
    else:
        print("error")
    return dataset