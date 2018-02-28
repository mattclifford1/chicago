import matplotlib.pyplot as plt
import numpy as np

m_train_20 = np.array([139.015887561,123.625739586,194.70993395,166.340866708])
m_test_20 = np.array([169.761505398,132.135649861,193.598432461,174.476104601])

m_train_50 = np.array([56.1928746585,44.5512765801,157.298942609,48.107303862])
m_test_50 = np.array([45.9722856845,44.9566170271,156.846756876,43.1813162422])

m_train_100 = np.array([125.151603475,83.8130269872,150.10080736,110.617766568])
m_test_100 = np.array([147.619247364,87.9708423003,150.509920122,114.814581606])

m_train_140 = np.array([56.9403400151,43.9913041315,63.184027485,45.7246228881])
m_test_140 = np.array([48.7111074746,43.3782099561,67.4685780473,43.9077375342])

m_train_150 = np.array([65.3566724299,82.8976963167,51.1534269366,50.6616493269])
m_test_150 = np.array([59.3335351673,86.0055449677,56.9966016174,48.962155744])

m_train_160 = np.array([50.8978639236,67.5227100409,68.7332903467,71.7097483479])
m_test_160 = np.array([29.2851609475,69.7907215479,72.8175178026,72.3222409725])

m_train_200 = np.array([65.3404389796,64.5840423687,125.37770711,79.7696842641])
m_test_200 = np.array([60.9724554866,65.4882905181,126.881355385,81.0536369855])

train = [m_train_20, m_train_50, m_train_100, m_train_140, m_train_150, m_train_160, m_train_200]
test = [m_test_20, m_test_50, m_test_100, m_test_140, m_test_150, m_test_160, m_test_200]

components = [20,50,100,140,150,160,200]
mean = [0]* len(train)
lower = [0]* len(train)
upper = [0]* len(train)
i= 0
for m in test:
	mean[i] = np.mean(m)
	lower[i] = np.min(m)
	upper[i] = np.max(m)
	i+=1
plt.errorbar(components, mean, yerr=[lower, upper],label='Test Data',capsize=5)
mean = [0]* len(train)
lower = [0]* len(train)
upper = [0]* len(train)
i= 0
for m in train:
	mean[i] = np.mean(m)
	lower[i] = np.min(m)
	upper[i] = np.max(m)
	i+=1
plt.errorbar(components, mean, yerr=[lower, upper],label='Train Data',capsize=5)
plt.xlabel('Number of Components')
plt.ylabel('Squared Error')
plt.legend()
plt.title('Cross Validation with data set split into 4 parts')
plt.show()
