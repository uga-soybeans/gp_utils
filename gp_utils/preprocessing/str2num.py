import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
import re

class str2numConverter(TransformerMixin):
    '''
    Convert genotype data into standardized numeric encoding {-1, 0, 1}.
    Supports numeric, A/H/B and allele-call encodings.
    '''
    def __init__(self, read_only=False):
        self.reference_alleles_ = {} # Will store per-column allele mapping rules after fitting
        self.encoding_type_ = None  # 'numeric_-101', 'numeric_012', 'AHB', 'allele_call'
        self.columns_ = None
        self.read_only = read_only # If True, only validates encoding types without recording and cannot perform transform

    def fit(self, X, y=None):
        """
        Learn allele mapping for each column if needed.
        Does not support monomorphic markers for unlabeled allele-call encoding.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        
        self.reference_alleles_ = {} # reset reference alleles
        if not self.read_only:
            self.columns_ = X.columns.tolist() # reset marker names

        sample_values = X.iloc[0:5].apply(lambda col: col.dropna().unique())

        # Detect encoding type automatically
        self.encoding_type_ = self._detect_encoding_type(sample_values)

        if not self.read_only and self.encoding_type_ == "allele_call_unlabeled":
            self._learn_reference_alleles(X)

        return self
    
    def transform(self, X):
        """
        Convert the dataframe into numeric genotype matrix.
        Returns a numpy array of shape (n_samples, n_markers).
        """
        if self.read_only:
            return None

        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        if self.encoding_type_ is None:
            raise RuntimeError("You must fit the encoder before transforming data.")
        
        if X.columns.tolist() != self.columns_:
            raise ValueError("Input X columns do not match with training data.")

        df = X.copy()
        if self.encoding_type_ == "numeric_-101":
            return df.to_numpy(dtype=float)
        elif self.encoding_type_ == "numeric_012":
            return (df.astype(float) - 1).to_numpy()
        elif self.encoding_type_ == "AHB":
            mapping = {"A": 1, "H": 0, "B": -1}
            return df.replace(mapping).to_numpy(dtype=float)
        elif self.encoding_type_ == "allele_call_labeled":
            return self._convert_allele_call_labeled(df)
        elif self.encoding_type_ == "allele_call_unlabeled":
            return self._convert_allele_call_unlabeled(df) # Warning: does not allow monomorphic markers
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type_}")

    # ------ Internal utilities ------
    def _detect_encoding_type(self, sample_values):
        """Heuristically detect encoding type from sample values."""
        flattened = np.unique(np.concatenate(sample_values.values))
        
        numeric_vals, numeric_012, ahb, allele_like = False, False, False, False

        if np.issubdtype(flattened.dtype, np.number):
            numeric_vals = ((flattened >= -1) & (flattened <= 1)).all()
            numeric_012 = ((flattened >= 0) & (flattened <= 2)).all()
        else:
            flattened = [str(v) for v in flattened if v != "nan"]
            ahb = set(flattened).issubset({"A", "H", "B"})
            allele_like = all(re.fullmatch(r"[ACGT]{2}", v) for v in flattened if v not in {"nan"})

        if numeric_vals:
            return "numeric_-101"
        elif numeric_012:
            return "numeric_012"
        elif ahb:
            return "AHB"
        elif allele_like:
            # Check if column names encode allele info
            if all(re.search(r"_[ACGT]_[ACGT]$", col) for col in sample_values.index):
                return "allele_call_labeled"
            else:
                return "allele_call_unlabeled"
        else:
            raise ValueError("Unable to detect genotype encoding type.")

    def _learn_reference_alleles(self, X):
        """
        For unlabeled allele calls, determine the reference allele per column
        (alphabetically first allele observed).
        Design choice: This function does not allow monomorphic markers
        """
        for col in X.columns:
            unique_vals = pd.Series(X[col].dropna().unique()).astype(str)
            alleles = set(allele for pair in unique_vals for allele in pair)
            if len(alleles) != 2:
                raise ValueError(f"Column {col} has invalid allele pattern: {alleles}. Please make sure all markers are biallelic and polymorphic.")
            ref, alt = sorted(list(alleles))
            self.reference_alleles_[col] = (ref, alt)


    def _convert_allele_call_labeled(self, df):
        """
        Convert columns with names like 'SNP1_A_T' -> AA=1, TT=-1, AT=0
        """
        out = np.zeros_like(df, dtype=float)
        for i, col in enumerate(df.columns):
            match = re.search(r"_([ACGT])_([ACGT])$", col)
            if not match:
                raise ValueError(f"Cannot extract alleles from column name: {col}")
            ref, alt = match.groups()
            mapping = {ref + ref: 1, alt + alt: -1, ref + alt: 0, alt + ref: 0}
            out[:, i] = df[col].replace(mapping).to_numpy(dtype=float)
        return out

    def _convert_allele_call_unlabeled(self, df):
        """
        Use reference alleles learned in fit().
        """
        out = np.zeros_like(df, dtype=float)
        for i, col in enumerate(df.columns):
            ref, alt = self.reference_alleles_[col]
            mapping = {ref + ref: 1, alt + alt: -1, ref + alt: 0, alt + ref: 0}
            out[:, i] = df[col].replace(mapping).to_numpy(dtype=float)
        return out
