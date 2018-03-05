import matplotlib.pyplot as plt
import numpy as np

m_train_20 = np.array([146.87292619418838, 145.19054914839126, 153.10109476821432, 127.3298589132841])
m_test_20 = np.array([186.81951662275776, 157.52839722204982, 151.84644801764793, 136.75290874719192])

m_train_50 = np.array([87.036957322247503, 68.8983258801685, 54.301837079709991, 68.695189575177949])
m_test_50 = np.array([107.57020888443947, 74.485273834823943, 58.151183305213209, 73.740259099744776])

m_train_100 = np.array([73.853767030501388, 53.755779144261084, 48.637156870858298, 60.670859620633969])
m_test_100 = np.array([85.399072912180728, 56.620143118257324, 53.160104691787751, 64.067121650180866])

m_train_120 = np.array([73.134252842821127, 51.949288024466931, 63.338439732842993, 42.254313980753032])
m_test_120 = np.array([81.182020250466877, 54.125105022110212, 67.309387069944336, 42.045362953549585])

# m_train_130 = np.array()
# m_test_130 = np.array()

m_train_140 = np.array([62.123809114736737, 42.503434264558464, 44.543040324451304, 42.075925985717667])
m_test_140 = np.array([64.043182874712031, 42.92233613137482, 49.941743612888985, 42.273593152438735])

# m_train_150 = np.array()
# m_test_150 = np.array()

m_train_160 = np.array([52.750918078392779, 44.78176372302616, 40.063620771824226, 41.965865539386101])
m_test_160 = np.array([47.823139954142853, 45.245653839445204, 46.404887550338231, 41.475636177694618])

# m_train_200 = np.array()
# m_test_200 = np.array()

# train = [m_train_20, m_train_50, m_train_100, m_train_120, m_train_130, m_train_140, m_train_150, m_train_160, m_train_200]
# test = [m_test_20, m_test_50, m_test_100, m_test_120, m_test_130, m_test_140, m_test_150, m_test_160, m_test_200]

train = [m_train_20, m_train_50, m_train_100, m_train_120, m_train_140,  m_train_160]
test = [m_test_20, m_test_50, m_test_100, m_test_120, m_test_140, m_test_160]

components = [20,50,100,120,130,140,150,160,200]
components = [20,50,100,120,140,160]
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
