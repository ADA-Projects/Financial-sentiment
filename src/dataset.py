import pandas as pd
from torch.utils.data import Dataset

class FinancialPhraseBank(Dataset):
    """
    PyTorch Dataset for FinancialPhraseBank CSV or DataFrame.

    Args:
        data (str or pandas.DataFrame): Path to CSV file or DataFrame with columns 'Sentence' and 'Sentiment'.
        tokenizer: HuggingFace tokenizer instance.
        max_length (int): Max token length for truncation/padding.

    Behavior:
        - Reads CSV if `data` is a file path; clones DataFrame if passed directly.
        - Validates required columns.
        - Converts string labels to integer IDs via an internal mapping.
        - Tokenizes on-the-fly in __getitem__ with padding/truncation.
    """
    def __init__(self, data, tokenizer, max_length: int = 128):
        # Load data
        if isinstance(data, str):
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("`data` must be a file path or pandas DataFrame")

        # Ensure necessary columns
        if not {'Sentence', 'Sentiment'}.issubset(df.columns):
            raise ValueError("DataFrame must contain 'Sentence' and 'Sentiment' columns")

        # Prepare texts
        self.texts = df['Sentence'].astype(str).tolist()

        # Prepare labels, mapping strings to ints if needed
        raw_labels = df['Sentiment']
        if raw_labels.dtype == object or isinstance(raw_labels.iloc[0], str):
            unique_vals = sorted(raw_labels.unique())
            self.label_map = {label: idx for idx, label in enumerate(unique_vals)}
            labels_mapped = raw_labels.map(self.label_map)
        else:
            self.label_map = None
            labels_mapped = raw_labels

        self.labels = labels_mapped.astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # Remove batch dimension
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item['labels'] = label
        return item

