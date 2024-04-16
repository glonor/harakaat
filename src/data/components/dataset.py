from torch.utils.data import Dataset


class TashkeelaDataset(Dataset):
    def __init__(self, tokenizer, dataset, split, max_len=512):
        self.split = split
        self.dataset_split = dataset[split]
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset_split)

    def __getitem__(self, index):
        row = self.dataset_split[index]

        target = row["diacratized"]
        text = row["text"]

        encoding = self.tokenizer.batch_encode_plus(
            [text], max_length=self.max_len, padding="max_length", return_tensors="pt"
        )
        target_encoding = self.tokenizer.batch_encode_plus(
            [target], max_length=self.max_len, padding="max_length", return_tensors="pt"
        )

        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
        target_ids = target_encoding.input_ids

        # replace padding token id's of the target_ids by -100 so it's ignored by the loss
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100

        # squeeze the tensors to remove the batch dimension
        input_ids = input_ids.squeeze()
        attention_mask = attention_mask.squeeze()
        target_ids = target_ids.squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
            "target": target,
        }
