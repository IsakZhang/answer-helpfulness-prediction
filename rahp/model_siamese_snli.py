from typing import Dict, List, Optional
from overrides import overrides

import torch
import torch.nn.functional as F

from allennlp.models.model import Model
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder, FeedForward
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("snli_siamese")
class SNLISiameseModel(Model):
    """
    This is a simple Siamese model to be trained on SNLI dataset, the main purpose is
        to transfer the trained weights to the RAHP model

    Network struture:
        embed - encode - cancatenate 2 vecs - make prediction
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 context_encoder: Seq2SeqEncoder,
                 text_encoder_entailment: Seq2VecEncoder,
                 matching_layer: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None
                ):
        super(SNLISiameseModel, self).__init__(vocab)

        self.text_field_embedder = text_field_embedder
        self.context_encoder = context_encoder
        self.text_encoder_entailment = text_encoder_entailment
        self.matching_layer = matching_layer

        # running value, reset the values every epoch
        self.accuracy = CategoricalAccuracy()
        self.criterion = torch.nn.CrossEntropyLoss()


    @overrides
    def forward(self, premise, hypothesis, label = None):
        
        # shape = (batch_size, seq_len)
        p_mask = get_text_field_mask(premise)
        h_mask = get_text_field_mask(hypothesis)
        
        # shape = (batch_size, seq_len, embed_dim)
        embedded_p = self.text_field_embedder(premise)
        embedded_h = self.text_field_embedder(hypothesis)
               
        # context encoder
        context_p = self.context_encoder(embedded_p, p_mask)
        context_h = self.context_encoder(embedded_h, h_mask)
        
        # inference encoder: encode to fixed-size vectors
        o_p = self.text_encoder_entailment(context_p, p_mask)
        o_h = self.text_encoder_entailment(context_h, h_mask)
        
        # feed to FC layer
        logits = self.matching_layer(torch.cat([o_p, o_h], dim=-1))
        probs = F.softmax(logits, dim=-1)
        output_dict = {'probs': probs}

        if label is not None:
            loss = self.criterion(logits, label.long().view(-1))
            self.accuracy(logits, label) 
            output_dict['loss'] = loss
            
        return output_dict

    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        metrics["accuracy"] = self.accuracy.get_metric(reset)
        return metrics