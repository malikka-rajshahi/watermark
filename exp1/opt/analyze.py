import torch
import torch.nn.functional as F

import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from transformers import MarianMTModel, MarianTokenizer

from collections import defaultdict
import pickle

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

'''
file = f'm35-run-1-trst.p'
results = pickle.load(open(file,"rb"))
print(results)
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x = [3, 5, 7, 10, 15, 20, 25, 30, 35, 40]
trst = []
print('trst.p')
for i in range(1, 11):
    # filename = f'm35-run-{i}-gued.p'
    # file = os.path.join(dir, filename)
    file = f'run-{i}-trst.p'
    results = pickle.load(open(file,"rb"))
    pvals = results['watermark']['pvals']
    if pvals.dim() == 0: pvals = pvals.unsqueeze(0)
    median = torch.mean(pvals.float())
    trst.append(median.item())
    print(f"Median p-value of watermarked completions for all prompts: {median}")

gust = []
print('gust')
for i in range(1, 11):
    # filename = f'm35-run-{i}-gued.p'
    # file = os.path.join(dir, filename)
    file = f'run-{i}-gust.p'
    results = pickle.load(open(file,"rb"))
    pvals = results['watermark']['pvals']
    if pvals.dim() == 0: pvals = pvals.unsqueeze(0)
    median = torch.mean(pvals.float())
    gust.append(median.item())
    print(f"Median p-value of watermarked completions for all prompts: {median}")

'''
tred = []
print('tred')
for i in range(1, 11):
    # filename = f'm35-run-{i}-gued.p'
    # file = os.path.join(dir, filename)
    file = f'run-{i}-tred.p'
    results = pickle.load(open(file,"rb"))
    pvals = results['watermark']['pvals']
    if pvals.dim() == 0: pvals = pvals.unsqueeze(0)
    median = torch.mean(pvals.float())
    tred.append(median.item())
    print(f"Median p-value of watermarked completions for all prompts: {median}")

gued = []
print('gued')
for i in range(1, 11):
    # filename = f'm35-run-{i}-gued.p'
    # file = os.path.join(dir, filename)
    file = f'run-{i}-gued.p'
    results = pickle.load(open(file,"rb"))
    pvals = results['watermark']['pvals']
    if pvals.dim() == 0: pvals = pvals.unsqueeze(0)
    median = torch.mean(pvals.float())
    gued.append(median.item())
    print(f"Median p-value of watermarked completions for all prompts: {median}")
'''

kirch10 = []
print('kirch')
for i in range(1, 11):
    # filename = f'm35-run-{i}-gued.p'
    # file = os.path.join(dir, filename)
    file = f'run-{i}-ki10.p'
    results = pickle.load(open(file,"rb"))
    pvals = results['watermark']['pvals']
    if pvals.dim() == 0: pvals = pvals.unsqueeze(0)
    median = torch.mean(pvals.float())
    kirch10.append(median.item())
    print(f"Median p-value of watermarked completions for all prompts: {median}")

kirch20 = []
print('kirch')
for i in range(1, 11):
    # filename = f'm35-run-{i}-gued.p'
    # file = os.path.join(dir, filename)
    file = f'run-{i}-ki20.p'
    results = pickle.load(open(file,"rb"))
    pvals = results['watermark']['pvals']
    if pvals.dim() == 0: pvals = pvals.unsqueeze(0)
    median = torch.mean(pvals.float())
    kirch20.append(median.item())
    print(f"Median p-value of watermarked completions for all prompts: {median}")

data = {
    'Fraction of substitutions': x,
    'ITS': trst,
    'EXP': gust,
    'KGW-1.0': kirch10,
    'KGW-2.0': kirch20
}
df = pd.DataFrame(data)

sns.set_theme(style="darkgrid")
plt.figure(figsize=(8, 6))
sns.lineplot(x='Fraction of substitutions', y='ITS', data=df, label='ITS')
sns.lineplot(x='Fraction of substitutions', y='EXP', data=df, label='EXP')
#sns.lineplot(x='Fraction of substitutions', y='ITS-edit', data=df, label='ITS-edit')
#sns.lineplot(x='Fraction of substitutions', y='EXP-edit', data=df, label='EXP-edit')
sns.lineplot(x='Fraction of substitutions', y='KGW-1.0', data=df, label='KGW-1.0')
sns.lineplot(x='Fraction of substitutions', y='KGW-2.0', data=df, label='KGW-2.0')
plt.xlabel('Text length (m)')
plt.ylabel('Median p-value')
plt.legend()
plt.savefig('exp1_opt_2a.png', bbox_inches='tight')

