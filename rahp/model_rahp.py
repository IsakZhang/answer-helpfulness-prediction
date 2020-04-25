from typing import Dict, List, Optional
from overrides import overrides

import torch
import torch.nn.functional as F

from allennlp.models.model import Model
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, FeedForward, Seq2SeqEncoder
from allennlp.modules import Attention, MatrixAttention
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum
from allennlp.training.metrics import Auc, F1Measure, CategoricalAccuracy


@Model.register("rahp")
class AnswerHelpfulPredictionModel(Model):
    """
    This is the implementation of the RAPH model proposed in the paper.
    Given a question, its answer and relevant reviews, predict whether 
        the answer is helpful or not.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 context_encoder: Seq2SeqEncoder,
                 qa_attention_module: MatrixAttention,
                 text_encoder_qa_matching: Seq2VecEncoder,
                 qa_matching_layer: FeedForward,
                 qr_attention_module: Attention,
                 text_encoder_ra_entailment: Seq2VecEncoder,
                 ra_matching_layer: FeedForward,
                 predict_layer: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None
                ):
        super(AnswerHelpfulPredictionModel, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.context_encoder = context_encoder
        self.qa_attention_module = qa_attention_module
        self.text_encoder_qa_matching = text_encoder_qa_matching
        self.qa_matching_layer = qa_matching_layer
        self.qr_attention_module = qr_attention_module
        self.text_encoder_ra_entailment = text_encoder_ra_entailment
        self.ra_matching_layer = ra_matching_layer
        self.predict_layer = predict_layer

        # performance scores are running values, reset the values every epoch
        self.f1_measure = F1Measure(positive_label=1) 
        self.auc_score = Auc(positive_label=1)
        self.accuracy = CategoricalAccuracy()
        
        self.criterion = torch.nn.CrossEntropyLoss()
        initializer(self)

        
    @overrides
    def forward(self, question, answer, reviews, helpful = None):

        # ----------------------------------------------------------
        # layer-1: Embed q/a/reviews and Encode context infomation

        # shape = (batch_size, seq_len)
        q_mask = get_text_field_mask(question)
        a_mask = get_text_field_mask(answer)
        # shape = (batch_size, 5, seq_len)
        r_mask = get_text_field_mask(reviews, num_wrapping_dims=1)
        
        # shape = (batch_size, seq_len, embed_dim)
        embedded_q = self.text_field_embedder(question)
        embedded_a = self.text_field_embedder(answer)
        # shape = (batch_size, 5, seq_len, embed_dim)
        embedded_r = self.text_field_embedder(reviews, num_wrapping_dims=1)
        review_ls = [(embedded_r[:,i,:,:], r_mask[:,i,:]) for i in range(5)]
        
        context_q = self.context_encoder(embedded_q, q_mask)
        context_a = self.context_encoder(embedded_a, a_mask)
        # shape of context_r[i]: (bs, seq_len, encoding_dim)
        context_r = [self.context_encoder(r[0], r[1]) for r in review_ls]
        

        # ----------------------------------------------------------
        # layer-2: QA Matching
        
        # shape = (bs, len_q, len_a)
        sim_matrix = self.qa_attention_module(context_q, context_a)
        # masked attention to remove those paddings in Q/A
        a2q_attention = masked_softmax(sim_matrix, a_mask)
        q2a_attention = masked_softmax(sim_matrix.transpose(1, 2).contiguous(), q_mask)
        
        # shape = (batch_size, len_q, encoding_dim)
        attended_q_from_a = weighted_sum(context_a, a2q_attention)
        attended_a_from_q = weighted_sum(context_q, q2a_attention)

        v_q = torch.cat([context_q, attended_q_from_a], dim=-1)
        v_a = torch.cat([context_a, attended_a_from_q], dim=-1)

        # encoding the sequence info to a fixed vector
        o_q = self.text_encoder_qa_matching(v_q, q_mask)
        o_a_q = self.text_encoder_qa_matching(v_a, a_mask)
        qa_matching_score = self.qa_matching_layer(torch.cat([o_q, o_a_q], dim=-1))
        

        # ----------------------------------------------------------
        # layer-3: RA coherence modeling
        
        # use question text to highlight relevant review info
        q_enhanced_r = []
        qr_attention_weights = []

        for i in range(5):
            # Q2R attention (vec-matrix attention)
            # shape = (bs, len_r_i)
            beta = self.qr_attention_module(o_q, context_r[i], review_ls[i][1])
            
            qr_attention_weights.append(beta) # for visulazation
          
            enhanced_r_i = torch.matmul(beta.unsqueeze(1), context_r[i]).squeeze(1)
            q_enhanced_r.append(enhanced_r_i)
        
        # for visulazation attention weights
        qr_attention_weights_op = torch.cat(qr_attention_weights, dim=-1)
            
        # v_a/v_r shape = (bs, encoding_shape)
        o_a_r = self.text_encoder_ra_entailment(context_a, a_mask)
        o_r = [self.text_encoder_ra_entailment(context_r[i], review_ls[i][1]) + q_enhanced_r[i] for i in range(5)]
        
        # "K entailment checking"
        entailment_result = [self.ra_matching_layer(torch.cat([o_a_r, o_r[i]], dim=-1)) for i in range(5)]    
        entailment_score = torch.cat(entailment_result, dim=1)
        

        # ----------------------------------------------------------
        # layer-4: Final prediction layer
        
        logits = self.predict_layer(torch.cat([qa_matching_score, entailment_score], dim=-1))
        probs = F.softmax(logits, dim=-1)
        output_dict = {'probs': probs, 'entailment_score': entailment_score, 
                       'qr_attention': qr_attention_weights_op, "qa_attention": sim_matrix}

        if helpful is not None:
            loss = self.criterion(logits, helpful)
            self.f1_measure(logits, helpful)
            self.auc_score(probs[:,1], helpful) 
            # self.accuracy(logits, helpful)
            output_dict['loss'] = loss
            
        return output_dict

    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        precision, recall, f1_measure = self.f1_measure.get_metric(reset)
        metrics["f1"] = f1_measure
        metrics["auc"] = self.auc_score.get_metric(reset)
        # metrics["accuracy"] = self.accuracy.get_metric(reset)
        return metrics