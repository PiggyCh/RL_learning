class arguments():
    def __init__(self):
        self.gamma = 0.99
        self.action_dim = 2
        self.obs_dim = (4,84,84) #80*80*3
        self.capacity = 50000
        self.cuda = 'cuda:0'
        self.Frames = 4
        self.episodes = int(1e8)
        self.updatebatch= 512
        self.test_episodes= 10
        self.epsilon = 0.1
        self.Q_NETWORK_ITERATION = 50
        self.learning_rate = 0.001