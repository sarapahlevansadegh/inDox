import torch
from transformers import BertTokenizer, BertForSequenceClassification
from ..utils import read_config
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import BERTScorer
from collections import defaultdict


class Evaluation:
    def __init__(self, dimansions):
        self.config = read_config()
        self.metrics = [eval(dim) for dim in dimansions]

    def __call__(self, inputs=None, *args, **kwargs):
        return self.metrics(inputs)

class BertScore(BERTScorer):
    def __init__(self):
        super().__init__()
    def calculate_score(self, inputs):
        _, answer, context = inputs['query'], inputs['answer'], inputs['context']
        scores = defaultdict(list)
        if not isinstance(context, list) and len(context) > 1:
            context = list(context)
        if not isinstance(answer, list):
            answer = [answer]
        P, R, F1 = self.score(answer, [context], verbose=False)
        return P, R, F1


class Toxicity:
    def __init__(self, cfg):
        self.bert_tokenizer = BertTokenizer.from_pretrained(cfg["evaluation"]["bert_toxic_tokenizer"])
        self.bert_model = BertForSequenceClassification.from_pretrained(cfg["evaluation"]["bert_toxic_model"])

    def calculate_toxicity(self, text):
        # Tokenizing input text and calculating toxicity
        inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        toxicity = probs[
            0, 1
        ].item()  # Assuming that the second class is the 'toxic' class
        return toxicity


class Similarity:
    def __init__(self, cfg, dims, inputs=None):
        self.reference = inputs["context"]
        self.candidate = inputs["answer"]
        # self.semantic_model = SentenceTransformer(cfg["evaluation"]["semantic_similarity"])
        self.vectorizer = TfidfVectorizer()
        self.mlb = MultiLabelBinarizer()

    def bleu_score(self) -> float:
        """
        Calculate the BLEU score between candidate and reference texts.
        Returns:
        float: The BLEU score.
        """

        reference = [word_tokenize(self.reference)]
        candidate = word_tokenize(self.candidate)

        return sentence_bleu(reference, candidate)

    def jaccard_similarity(self) -> float:
        """
        Calculate the Jaccard similarity between candidate and reference texts.
        Returns:
        float: The Jaccard similarity score.
        """
        reference_tokens = set(word_tokenize(self.reference))
        model_output_tokens = set(word_tokenize(self.candidate))

        binary_reference = self.mlb.fit_transform([reference_tokens])
        binary_model_output = self.mlb.transform([model_output_tokens])

        return jaccard_score(
            binary_reference[0], binary_model_output[0], average="binary"
        )

    def cosine_similarity(self) -> float:
        """
        Calculate the cosine similarity between the TF-IDF vectors of the candidate and reference texts.
        Returns:
        float: The cosine similarity score.
        """
        vectors = self.vectorizer.fit_transform([self.candidate, self.reference])
        return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    def semantic_similarity(self) -> float:
        """
        Calculate the semantic similarity between candidate and reference texts using BERT embeddings.

        Returns:
        float: The semantic similarity score.
        """

        embeddings = self.semantic_model.encode(
            [self.candidate, self.reference], convert_to_tensor=True
        )
        return util.pytorch_cos_sim(embeddings[0], embeddings[1])[0][0].item()
