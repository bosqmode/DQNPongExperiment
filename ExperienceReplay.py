from collections import deque
import random

class ExperienceReplay:
	def __init__(self, maxlen):
		self.maxlen = maxlen
		self.samples = deque(maxlen=maxlen)

	def Length(self):
		return len(self.samples)

	def Append(self, sample):
		self.samples.append(sample)

	def RandomBatch(self, batchsize):
		batch = min(batchsize, self.Length())
		return random.sample(self.samples, batch)