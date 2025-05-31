#!/usr/bin/env python3
"""
Investigate the training data to understand why predictions are wrong
"""

import pandas as pd
import numpy as np
from collections import Counter

def investigate_data():
    """Analyze the training data to find issues"""
    
    print("🔍 Investigating Financial Phrase Bank Data...")
    print("=" * 60)
    
    # Load original data
    df = pd.read_csv('data/data.csv')
    
    print(f"📊 Original Dataset: {len(df)} samples")
    print(f"📊 Class distribution:")
    class_counts = df['Sentiment'].value_counts()
    for sentiment, count in class_counts.items():
        percentage = 100 * count / len(df)
        print(f"   {sentiment}: {count} ({percentage:.1f}%)")
    
    print("\n" + "=" * 60)
    print("🔍 Sample sentences by sentiment:")
    
    # Show examples from each class
    for sentiment in ['negative', 'neutral', 'positive']:
        print(f"\n📝 {sentiment.upper()} examples:")
        examples = df[df['Sentiment'] == sentiment]['Sentence'].head(5).tolist()
        for i, example in enumerate(examples, 1):
            print(f"   {i}. {example}")
    
    print("\n" + "=" * 60)
    print("🔍 Looking for confusing examples...")
    
    # Look for potentially mislabeled examples
    positive_keywords = ['profit', 'gain', 'growth', 'increase', 'rise', 'strong', 'beat', 'soar']
    negative_keywords = ['loss', 'decline', 'fall', 'drop', 'crash', 'weak', 'miss', 'plunge']
    
    # Find negative sentences with positive keywords
    print("\n🤔 Negative sentences with positive keywords:")
    negative_df = df[df['Sentiment'] == 'negative']
    for idx, row in negative_df.iterrows():
        sentence = row['Sentence'].lower()
        if any(keyword in sentence for keyword in positive_keywords):
            print(f"   • {row['Sentence']}")
            if len([s for s in negative_df['Sentence'] if any(k in s.lower() for k in positive_keywords)]) >= 5:
                break
    
    # Find positive sentences with negative keywords  
    print("\n🤔 Positive sentences with negative keywords:")
    positive_df = df[df['Sentiment'] == 'positive']
    for idx, row in positive_df.iterrows():
        sentence = row['Sentence'].lower()
        if any(keyword in sentence for keyword in negative_keywords):
            print(f"   • {row['Sentence']}")
            if len([s for s in positive_df['Sentence'] if any(k in s.lower() for k in negative_keywords)]) >= 5:
                break
    
    print("\n" + "=" * 60)
    print("🔍 Analyzing balanced dataset (what model actually sees)...")
    
    # Simulate the balancing process
    label_counts = df['Sentiment'].value_counts()
    max_count = label_counts.max()
    
    balanced_dfs = []
    for label in df['Sentiment'].unique():
        label_df = df[df['Sentiment'] == label]
        upsampled = label_df.sample(max_count, replace=True, random_state=42)
        balanced_dfs.append(upsampled)
    
    balanced_df = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"📊 Balanced Dataset: {len(balanced_df)} samples")
    balanced_counts = balanced_df['Sentiment'].value_counts()
    for sentiment, count in balanced_counts.items():
        percentage = 100 * count / len(balanced_df)
        print(f"   {sentiment}: {count} ({percentage:.1f}%)")
    
    # Check if we're duplicating bad examples
    print(f"\n📊 Duplication analysis:")
    for sentiment in ['negative', 'neutral', 'positive']:
        original_count = len(df[df['Sentiment'] == sentiment])
        balanced_count = len(balanced_df[balanced_df['Sentiment'] == sentiment])
        duplication_factor = balanced_count / original_count
        print(f"   {sentiment}: {duplication_factor:.1f}x duplication")
    
    print("\n" + "=" * 60)
    print("💡 Recommendations:")
    
    if class_counts.min() / class_counts.max() < 0.3:
        print("1. ⚠️  Severe class imbalance detected")
        print("2. 🔧 Try different balancing: undersample majority instead")
        print("3. 🔧 Use class weights instead of upsampling")
        print("4. 🔧 Try focal loss for hard examples")
    
    # Check if negative class is too small
    neg_ratio = class_counts['negative'] / len(df)
    if neg_ratio < 0.2:
        print(f"5. ⚠️  Negative class very small ({neg_ratio:.1%})")
        print("6. 🔧 Consider collecting more negative examples")
    
    return df, balanced_df

def analyze_wrong_predictions():
    """Analyze why specific predictions are wrong"""
    
    print("\n" + "=" * 60)
    print("🔍 Manual Analysis of Wrong Predictions:")
    
    wrong_cases = [
        {
            "sentence": "Company profits soared 25% this quarter",
            "predicted": "negative",
            "expected": "positive",
            "analysis": "Should be clearly positive - 'profits soared' is good news"
        },
        {
            "sentence": "Tesla stock price jumps on strong delivery numbers", 
            "predicted": "negative",
            "expected": "positive", 
            "analysis": "Should be positive - 'jumps' and 'strong delivery' are good"
        },
        {
            "sentence": "Market crash threatens investor portfolios",
            "predicted": "neutral", 
            "expected": "negative",
            "analysis": "Should be negative - 'crash' and 'threatens' are bad"
        }
    ]
    
    for case in wrong_cases:
        print(f"\n📝 '{case['sentence']}'")
        print(f"   Expected: {case['expected']}")
        print(f"   Predicted: {case['predicted']}")
        print(f"   Analysis: {case['analysis']}")

if __name__ == "__main__":
    df_original, df_balanced = investigate_data()
    analyze_wrong_predictions()