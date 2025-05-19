import pandas as pd
from torch.utils.data import Dataset


class FinancialPhraseBank(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=128):
        


        df = pd.read_csv(csv_path)
        # ensure columns: Sentence, Sentiment (0=neg,1=neu,2=pos)
        self.texts = df['Sentence'].tolist()
        self.labels = df['Sentiment'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True, padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding.input_ids.squeeze(),
            'attention_mask': encoding.attention_mask.squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
