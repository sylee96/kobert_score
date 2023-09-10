import os
import torch

from collections import Counter, defaultdict
from itertools import chain
from math import log
from typing import Union
from transformers import PreTrainedTokenizer, PreTrainedModel

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def sents_to_tensor(
        tokenizer: PreTrainedTokenizer,
        sents: Union[list[str], str],
        max_length: int = 512,
) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """
        Convert sentences into tensors using a tokenizer for pre-trained models.

        Args:
            tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenizing input sentences.
            sents (Union[list[str], str]): Either a list of sentences or a single sentence to tokenize.
            max_length (int): The maximum length for tokenized sequences (default is 512).

        Returns:
            tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]: A tuple of LongTensor containing:
                - input_ids: Token IDs of the input sentences.
                - attention_mask: Attention mask indicating token positions.
                - token_mask: Token mask excluding special tokens (CLS and SEP).

        Example:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            sentences = ["이것은 예시문장입니다.", "또 다른 예시 문장입니다."]
            input_ids, attention_mask, token_mask = sents_to_tensor(tokenizer, sentences)

            # The function converts input sentences into tensors suitable for model input.
        """
    # Tokenize sentences, create input tensors, and calculate token mask.
    # The function returns three tensors: input_ids, attention_mask, and token_mask.

    inputs = tokenizer(sents, padding=True, truncation=True, max_length=max_length)
    input_ids = torch.LongTensor(inputs['input_ids'])
    attention_mask = torch.LongTensor(inputs['attention_mask'])

    zero_mask = torch.zeros(attention_mask.size(), dtype=torch.long)
    token_mask = torch.where(input_ids == tokenizer.cls_token_id, zero_mask, attention_mask)
    token_mask = torch.where(input_ids == tokenizer.sep_token_id, zero_mask, token_mask).long()

    return input_ids, attention_mask, token_mask


def get_contextual_embeddings(
        model: PreTrainedModel,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
) -> torch.tensor:
    """
    Obtain contextual embeddings from a pre-trained language model.

    Args:
        model (PreTrainedModel): The pre-trained language model (e.g., BERT, RoBERTa, ELECTRA).
        input_ids (torch.LongTensor): Tensor containing token IDs for the input text.
        attention_mask (torch.LongTensor): Tensor containing attention mask for input tokens.

    Returns:
        torch.Tensor: Contextual embeddings for the input obtained from the model's last hidden state.

    Example:
        model = AutoModel.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        input_text = "날씨가 많이 춥다"
        input_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0)  # Add batch dimension
        attention_mask = torch.ones_like(input_ids)

        embeddings = get_contextual_embeddings(model, input_ids, attention_mask)

        # The function retrieves contextual embeddings from the model.
    """
    # Check for GPU availability and move tensors accordingly.
    # Compute contextual embeddings from the model's last hidden state.

    if torch.cuda.is_available():
        device = torch.cuda()
    else:
        device = torch.device('cpu')

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    contextual_embeds = outputs.last_hidden_state

    return contextual_embeds


def _count_doc_freq(
        inputs: tuple,
) -> set:
    """
    Compute the document frequency of tokens in the input_ids tensor based on the weight_mask.

    Args:
        inputs (tuple): A tuple containing two tensors, input_ids and weight_mask.
                        input_ids (torch.Tensor): Tensor containing token IDs.
                        weight_mask (torch.Tensor): Binary mask indicating which tokens to consider.

    Returns:
        set: A set containing the unique tokens that have a weight (1) in the weight_mask.

    Example:
        input_ids = torch.tensor([0, 3822, 5792, 2259, 1677, 2062, 18, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        weight_mask = torch.tensor([0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        inputs = (input_ids, weight_mask)
        doc_freq = _count_doc_freq(inputs)
        # Resulting doc_freq set: {3822, 5792, 2259, 1677, 2062, 18}
    """
    input_ids, weight_mask = inputs

    # Extract tokens from input_ids where weight_mask is 1 (indicating tokens to consider)
    tokens = input_ids[weight_mask == 1].tolist()

    # Create a set to store unique tokens with document frequency
    doc_freq = set(tokens)

    return doc_freq


def get_idf_weights(
        input_ids: torch.LongTensor,
        weight_mask: torch.LongTensor,
) -> dict:
    """
    Compute IDF (Inverse Document Frequency) weights for tokens based on input_ids and weight_mask.

    Args:
        input_ids (torch.LongTensor): Tensor containing token IDs.
        weight_mask (torch.LongTensor): Binary mask indicating which tokens to consider.

    Returns:
        dict: A dictionary containing IDF weights for tokens.

    Example:
        input_ids = torch.tensor([[0, 3822, 5792, 2259, 1677, 2062, 18, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [0, 4892, 2079, 4438, 2259, 3671, 28674, 18, 2, 1, 1, 1, 1, 1, 1, 1, 1]])
        weight_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        idf_weights = get_idf_weights(input_ids, weight_mask)
        # Resulting idf_weights dictionary: {3822: 4.3829, 5792: 1.2393, 2259: 0.0, 1677: 2.3957, 2062: 1.0983, 18: 0.0, ...}

    Notes:
        This function computes IDF weights for tokens based on their document frequency. It first counts the document
        frequency of each token, then calculates the IDF weight for each token using the formula:
        IDF(token) = log((num_docs + 1) / (document frequency of token + 1))

    """
    idf_count = Counter()
    num_docs = input_ids.size()[0]

    # Count document frequency of tokens using the _count_doc_freq function
    idf_count.update(chain.from_iterable(map(_count_doc_freq, zip(input_ids, weight_mask))))

    # Initialize an IDF dictionary with a default value
    idf_dict = defaultdict(lambda: log((num_docs + 1) / 1))

    # Calculate IDF weights for tokens based on document frequency
    idf_dict.update(
        {idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()}
    )

    return idf_dict


def _convert_ids_to_idf_weights(
        input_ids: torch.LongTensor,
        weight_mask: torch.LongTensor,
        idf_dict: dict,
) -> torch.tensor:
    masked_input_ids = torch.where(weight_mask == 1, input_ids, -1)
    idf_weights = torch.tensor([[idf_dict[int(ids)] if int(ids) != -1 else 0.0 for ids in sent] for sent in masked_input_ids])
    return idf_weights


def _convert_ids_to_idf_weights(
        input_ids: torch.LongTensor,
        weight_mask: torch.LongTensor,
        idf_dict: dict,
) -> torch.tensor:
    """
    Convert input_ids to IDF (Inverse Document Frequency) weights based on the provided idf_dict.

    Args:
        input_ids (torch.LongTensor): Tensor containing token IDs.
        weight_mask (torch.LongTensor): Binary mask indicating which tokens to consider.
        idf_dict (dict): A dictionary containing IDF weights for tokens.

    Returns:
        torch.tensor: A tensor containing IDF weights corresponding to input_ids.

    Example:
        input_ids = torch.tensor([[0, 3822, 5792, 2259, 1677, 2062, 18, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [0, 4892, 2079, 4438, 2259, 3671, 28674, 18, 2, 1, 1, 1, 1, 1, 1, 1, 1]])
        weight_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        idf_weights = {...}  # A dictionary containing IDF weights for tokens
        converted_weights = _convert_ids_to_idf_weights(input_ids, weight_mask, idf_weights)
        # Resulting converted_weights tensor: Same shape as input_ids with IDF weights for tokens.
        idf_weights = torch.tensor([[0.0, 1.2949, 2.2982, 1.2904, 3.3902, 2.2940, 0.2902, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0],
                                 [0.0, 2.3892, 3.4902, 1.4909, 3.3899, 1.9298, 3.5902, 0.2902, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0]])

    Notes:
        This function converts input_ids to IDF weights based on the provided idf_dict. It replaces token IDs in
        input_ids with their corresponding IDF weights from idf_dict while keeping tokens indicated by 1 of weight_mask
        as is.
    """
    masked_input_ids = torch.where(weight_mask == 1, input_ids, -1)

    # Convert token IDs to IDF weights using idf_dict
    idf_weights = torch.tensor(
        [[idf_dict[int(ids)] if int(ids) != -1 else 0.0 for ids in sent] for sent in masked_input_ids]
    )

    return idf_weights


def get_bert_score(
        pairwise_cosine_similarity: torch.Tensor,
        importance_weighting: bool,
        refs_ids: torch.LongTensor,
        cands_ids: torch.LongTensor,
        refs_weight_mask: torch.LongTensor,
        cands_weight_mask: torch.LongTensor,
        refs_idf_dict: dict,
        cands_idf_dict: dict,
        baseline_rescaling: float,
) -> tuple[list, list, list]:
    """
    Compute the BERTScore for a pair of reference and candidate sentences.

    Args:
        pairwise_cosine_similarity (torch.Tensor): Tensor of pairwise cosine similarities between tokens.
        importance_weighting (bool): Flag indicating whether to use IDF-based importance weighting.
        refs_ids (torch.LongTensor): Tensor containing token IDs for reference sentences.
        cands_ids (torch.LongTensor): Tensor containing token IDs for candidate sentences.
        refs_weight_mask (torch.LongTensor): Binary mask indicating which tokens in references to consider.
        cands_weight_mask (torch.LongTensor): Binary mask indicating which tokens in candidates to consider.
        refs_idf_dict (dict): A dictionary containing IDF weights for reference sentence tokens.
        cands_idf_dict (dict): A dictionary containing IDF weights for candidate sentence tokens.
        baseline_rescaling (float): Value for baseline rescaling.

    Returns:
        tuple: A tuple containing the BERTScore metrics (R_BERTScore, P_BERTScore, F_BERTScore).

    Notes:
        This function computes the BERTScore for a pair of reference and candidate sentences based on
        the pairwise cosine similarity between tokens. It allows for IDF-based importance weighting
        if `importance_weighting` is True. The BERTScore is computed using precision (P_BERTScore)
        and recall (R_BERTScore), and their harmonic mean (F_BERTScore).
    """
    r_max, _ = pairwise_cosine_similarity.max(dim=-1)
    p_max, _ = pairwise_cosine_similarity.max(dim=1)

    if importance_weighting:
        refs_idf_weights = _convert_ids_to_idf_weights(refs_ids, refs_weight_mask, refs_idf_dict)
        cands_idf_weights = _convert_ids_to_idf_weights(cands_ids, cands_weight_mask, cands_idf_dict)
    else:
        refs_idf_weights = refs_weight_mask
        cands_idf_weights = cands_weight_mask

    R_BERTScore = (r_max * refs_idf_weights).sum(axis=-1) / refs_idf_weights.sum(axis=-1)
    P_BERTScore = (p_max * cands_idf_weights).sum(axis=-1) / cands_idf_weights.sum(axis=-1)
    F_BERTScore = 2 * (R_BERTScore + P_BERTScore) / (R_BERTScore + P_BERTScore)

    R_BERTScore = ((R_BERTScore - baseline_rescaling) / (1 - baseline_rescaling)).tolist()
    P_BERTScore = ((P_BERTScore - baseline_rescaling) / (1 - baseline_rescaling)).tolist()
    F_BERTScore = ((F_BERTScore - baseline_rescaling) / (1 - baseline_rescaling)).tolist()

    return (R_BERTScore, P_BERTScore, F_BERTScore)
