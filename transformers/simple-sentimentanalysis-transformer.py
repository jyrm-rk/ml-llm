##https://www.youtube.com/watch?v=QEaBAZQCtwE&ab_channel=AssemblyAI

from transformers import pipeline

classifiers = pipeline( task="sentiment-analysis")

res = classifiers("I love AI programming!!")

print(res)