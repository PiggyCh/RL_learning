import numpy as np
class Replay_Buffer():
    def __init__(self,arg):
        self.capacity=arg.capacity
        self.action_dim=arg.action_dim
        self.env_obs_space = arg.obs_dim
        self.data={
            'action':np.zeros((self.capacity,1)),
            'obs':np.zeros((self.capacity,self.env_obs_space[0],self.env_obs_space[1],self.env_obs_space[2])),
            'next_obs': np.zeros((self.capacity,self.env_obs_space[0],self.env_obs_space[1],self.env_obs_space[2])),
            'done':np.zeros((self.capacity,1)),
            'reward':np.zeros((self.capacity,1)),
        }
        self.ptr = 0
        self.isfull = 0
    def store_data(self, transition,length):
        if self.ptr+length>self.capacity:
            rest = self.capacity-self.ptr
            for key in self.data:
                store_tmp = np.array(transition[key][:],dtype=object)
                store_tmp=np.expand_dims(store_tmp,-1) if len(store_tmp.shape)== 1 else store_tmp
                self.data[key][self.ptr:]=store_tmp[:rest] #judge_edge
                transition[key]=transition[key][rest:]
            self.ptr=0
            length-=rest
            self.isfull=1
        for key in self.data:
            store_tmp = np.array(transition[key][:],dtype=object)
            self.data[key][self.ptr:self.ptr+length]= np.expand_dims(store_tmp,-1) if len(store_tmp.shape)== 1 else store_tmp #judge_edge
        self.ptr += length
    def sample(self,batch):
        if self.isfull:
            batch_index = np.random.choice(self.capacity,size=batch)
        else:
            batch_index = np.random.choice(self.ptr,size=batch)
        return {key:self.data[key][batch_index,:] for key in self.data}
