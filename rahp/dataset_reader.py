from typing import Iterator, Dict, List
import pickle
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, LabelField, TextField, ListField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


@DatasetReader.register("qa_review")
class QAReviewDatasetReader(DatasetReader):
    """
    This is the dataset reader used in the paper.
    It reads QA pairs and relevant reviews, and creates a dataset suitable 
    for predicting the answer helpfulness
    
    The output of ``read`` is a list of Instances with the following fields:
        question, answer: TextField
        reviews: ListField (a list of TextField)
        helpful: LabelField
    """
    def __init__(self, lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        
        super().__init__(lazy=False)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        

    @overrides
    def _read(self, file_path) -> Iterator[Instance]:
        """
        Read data from disk and produces a stream of `Instance`s
        """
        data = pickle.load(open(file_path, "rb"))
        for qa in data:
            helpful_score = qa['helpful'][0] / qa['helpful'][1]
            helpful = 1 if helpful_score == 1.0 else 0
            question = qa['questionText']
            answer = qa['answerText']
            reviews = [r for r in qa['reviews'][:5]]
            yield self.text_to_instance(question, answer, reviews, helpful)    


    @overrides
    def text_to_instance(self, question: str, answer: str, 
                         reviews: List[str], helpful: int = None) -> Instance:
        
        fields: Dict[str, Field] = {}
        
        tokenized_q = self._tokenizer.tokenize(question)
        tokenized_a = self._tokenizer.tokenize(answer)
        fields['question'] = TextField(tokenized_q, self._token_indexers)
        fields['answer'] = TextField(tokenized_a, self._token_indexers)

        fields['reviews'] = ListField([TextField(review, self._token_indexers)
                                       for review in self._tokenizer.batch_tokenize(reviews)])
    
        if helpful != None:
            fields['helpful'] = LabelField(helpful, skip_indexing=True)
        
        return Instance(fields)
