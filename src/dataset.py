import pandas as pd
from torch.utils.data import Dataset

class FinancialPhraseBank(Dataset):
    """
    PyTorch Dataset for FinancialPhraseBank CSV or DataFrame.

    Args:
        data (str or pandas.DataFrame): Path to CSV file or DataFrame with columns 'Sentence' and 'Sentiment'.
        tokenizer: HuggingFace tokenizer instance.
        max_length (int): Max token length for truncation/padding.
        label2id (dict, optional): Mapping from string labels to integer IDs. If not provided, one will be built from the data.

    Behavior:
        - Reads CSV if `data` is a file path; clones DataFrame if passed directly.
        - Validates required columns.
        - Uses `label2id` if given; otherwise builds a sorted-label mapping.
        - Stores `id2label` inverse mapping for inference.
        - Tokenizes on-the-fly in __getitem__ with padding/truncation.
    """
    def __init__(self, data, tokenizer, max_length: int = 128, label2id: dict = None):
        # Load DataFrame
        if isinstance(data, str):
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("`data` must be a file path or pandas DataFrame")

        # Validate columns
        if not {'Sentence', 'Sentiment'}.issubset(df.columns):
            raise ValueError("DataFrame must contain 'Sentence' and 'Sentiment' columns")

        # Store text
        self.texts = df['Sentence'].astype(str).tolist()

        # Label mapping
        raw_labels = df['Sentiment']
        if label2id is not None:
            self.label2id = label2id
            self.id2label = {v: k for k, v in label2id.items()}
            labels_mapped = raw_labels.map(self.label2id)
        else:
            # Build mapping from sorted unique string labels
            unique = sorted(raw_labels.unique())
            self.label2id = {l: i for i, l in enumerate(unique)}
            self.id2label = {i: l for l, i in self.label2id.items()}
            labels_mapped = raw_labels.map(self.label2id)

        # Final labels
        self.labels = labels_mapped.astype(int).tolist()

        # Tokenizer and length
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
        # Squeeze batch dim
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item['labels'] = label
        return item


