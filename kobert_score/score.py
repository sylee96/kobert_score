import jsonlines
import random

from tqdm import tqdm
from torch import linalg as LA
from transformers import AutoTokenizer, AutoModel
from utils import *


def _load_datas(
        data_path: str,
) -> list:
    """
    Load data from a specified file format (e.g., .txt or .jsonl).

    Args:
        data_path (str): The path to the data file.

    Returns:
        list: A list containing the loaded sentences from the file.

    This function reads data from a file and extracts sentences based on the provided file format.
    Supported formats include .txt and .jsonl. The function reads text lines from a .txt file or
    extracts 'text' values from a .jsonl file and stores them in a list.

    Example:
        data_path = "data/sample.txt"
        sentences = load_datas(data_path)

        # The function loads sentences from the specified file and returns them as a list.
    """

    sentences = list()
    file_name, file_extension = os.path.basename(data_path)[-1].split('.')

    if file_extension == 'txt':
        with open(data_path, 'r', encoding='utf-8') as reader:
            lines = reader.readlines()
            for line in tqdm(lines):
                line = line.strip()
                if line and line != '':
                    sentences.append(line)
    elif file_extension == 'jsonl':
        with jsonlines.open(data_path) as reader:
            for obj in tqdm(reader):
                if isinstance(obj['text'], str):
                    sentences.append(obj['text'])
    else:
        print(f"The {file_extension} format is not supported.")

    return sentences


def score(
        pretrained_model_name_or_path: str,
        data_path: str = None,
        refs: Union[list[str], str] = None,
        cands: Union[list[str], str] = None,
        importance_weighting: bool = True,
        baseline_rescaling: float = 0,
        verbose=None,
) -> tuple[list, list, list]:
    """
    Compute BERTScore between reference and candidate sentences using a pre-trained BERT model.

    Args:
        pretrained_model_name_or_path (str): Name or path of the pre-trained BERT model.
        data_path (str, optional): Path to a file containing sentences for computing BERTScore. Defaults to None.
        refs (Union[list[str], str], optional): List of reference sentences or path to a file containing them. Defaults to None.
        cands (Union[list[str], str], optional): List of candidate sentences or path to a file containing them. Defaults to None.
        importance_weighting (bool, optional): Flag indicating whether to use IDF-based importance weighting. Defaults to True.
        baseline_rescaling (float, optional): Rescaling factor for baseline BERTScore. Defaults to 0.

    Returns:
        tuple: A tuple containing the BERTScore metrics (R_BERTScore, P_BERTScore, F_BERTScore).

    Raises:
        AssertionError: If the number of reference sentences and candidate sentences are not equal.

    Example:
        pretrained_model_name_or_path = "bert-base-uncased"
        refs = ["Reference sentence 1", "Reference sentence 2"]
        cands = ["Candidate sentence 1", "Candidate sentence 2"]
        scores = score(pretrained_model_name_or_path, refs=refs, cands=cands)
        # Resulting scores: (R_BERTScore, P_BERTScore, F_BERTScore)

    Notes:
        This function computes the BERTScore between reference and candidate sentences using a pre-trained BERT model.
        It allows for importance weighting using Inverse Document Frequency (IDF) if `importance_weighting` is True.
        The BERTScore is computed using precision (P_BERTScore) and recall (R_BERTScore), and their harmonic mean (F_BERTScore).
    """

    if refs is not None and cands is not None:
        for ref in refs:
            assert isinstance(ref, str)

        for cand in cands:
            assert isinstance(cand, str)

        assert len(cands) == len(refs), \
            'Number of reference sentences and number of candidate sentences are not equal.'

    if refs is None:
        raise NotImplementedError(
            'There is no reference sentences!'
        )
    if cands is None:
        raise NotImplementedError(
            'There is no candidate sentences!'
        )

    if refs is None and cands is None and data_path is not None:
        sentences = _load_datas(data_path)
        random.shuffle(sentences)
        if len(sentences) < 20000:
            if len(sentences) % 2 == 1:
                sentences = sentences[:-1]
        else:
            sentences = sentences[:20000]

        refs = sentences[:len(sentences) // 2]
        cands = sentences[len(sentences) // 2:]

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    model = AutoModel.from_pretrained(pretrained_model_name_or_path)

    # sentence to tensor
    refs_ids, refs_attention_mask, refs_weight_mask = sents_to_tensor(tokenizer, refs)
    cands_ids, cands_attention_mask, cands_weight_mask = sents_to_tensor(tokenizer, cands)

    # tensor to contextual embeddings
    refs_embeds = get_contextual_embeddings(model, refs_ids, refs_attention_mask)
    cands_embeds = get_contextual_embeddings(model, cands_ids, cands_attention_mask)

    # Layer normalization
    refs_l2_norms = LA.vector_norm(refs_embeds, dim=-1, keepdim=True)
    cands_l2_norms = LA.vector_norm(cands_embeds, dim=-1, keepdim=True)
    normalized_refs_embeds = refs_embeds.div(refs_l2_norms)
    normalized_cands_embeds = cands_embeds.div(cands_l2_norms)

    # get pairwise cosine similarity
    pairwise_cosine_similarity = torch.bmm(normalized_refs_embeds, normalized_cands_embeds.transpose(1, 2))

    # # remove CLS token and SEP token
    # pairwise_cosine_similarity = pairwise_cosine_similarity[:, 1:-1, 1:-1]

    # get_idf_weights
    refs_idf_dict = get_idf_weights(refs_ids, refs_weight_mask)
    cands_idf_dict = get_idf_weights(cands_ids, cands_weight_mask)

    # compute Recall_BERT, Precision_BERT, F1_BERT
    R_BERTScore, P_BERTScore, F_BERTScore = get_bert_score(
        pairwise_cosine_similarity=pairwise_cosine_similarity,
        importance_weighting=importance_weighting,
        refs_ids=refs_ids,
        cands_ids=cands_ids,
        refs_weight_mask=refs_weight_mask,
        cands_weight_mask=cands_weight_mask,
        refs_idf_dict=refs_idf_dict,
        cands_idf_dict=cands_idf_dict,
        baseline_rescaling=baseline_rescaling,
    )

    return R_BERTScore, P_BERTScore, F_BERTScore
