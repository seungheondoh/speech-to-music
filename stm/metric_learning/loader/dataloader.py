from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, BatchSampler
from stm.metric_learning.loader.dataset import Emotion_Dataset

def get_dataloader(args, split):
    dataset = get_dataset(
        split = split,
        modality_type = args.modality, 
        speech_type = args.speech_type
    )
    if split == "TRAIN":
        data_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True,
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
        split,
        modality_type, 
        speech_type
    ):
    dataset = Emotion_Dataset(split, modality_type, speech_type)
    return dataset