#!/usr/bin/env python
# coding: utf-8


from sentence_transformers import SentenceTransformer,losses, models, util

import pandas as pd
from sacrebleu import corpus_bleu, corpus_chrf, sentence_bleu, sentence_chrf
import numpy as np
from tqdm import tqdm
from typing import Dict, List

model_save_path = 'xlsim/xlm-roberta-base-en-de'
metric_model = SentenceTransformer(model_save_path)

#Compute embedding for both lists
mt_samples = ['This is a mt sentence1','This is a mt sentence2']
ref_samples = ['This is a ref sentence1','This is a ref sentence2']
mtembeddings = metric_model.encode(mt_samples, convert_to_tensor=True)
refembeddings = metric_model.encode(ref_samples, convert_to_tensor=True)

#Compute cosine-similarities
cosine_scores_refmt = util.cos_sim(mtembeddings, refembeddings)
#cosine_scores_srcmt = util.cos_sim(mtembeddings, srcembeddings) #qe
metric_model_scores = []
for i in range(len(mt_samples)):
    metric_model_scores.append(cosine_scores_refmt[i][i].tolist())

scores = metric_model_scores
