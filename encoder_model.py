seed_value=42

# import the libraries
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

import random
random.seed(seed_value)

import numpy as np
np.random.seed(seed_value)

import tensorflow as tf
tf.random.set_seed(seed_value)



print("This Code was built using tensorflow version >={}".format(2.0))


def scaled_dot_attention(k,q,v,mask):
    
    '''
    Params: key,query & value matrix along with padded_mask or a combination of padded_mask+look_ahead_mask(broadcasted)
    shape(k):  (batch_size,num_attn_head,seq_len_k,dim_k)
    shape(q):  (batch_size,num_attn_head,seq_len_k,dim_q)
    shape(v):  (batch_size,num_attn_head,seq_len_k,dim_v)
    
    assert dim_k==dim_q in order to do matrix multiplication
    
    shape(mask):(batch_size,1,1,seq_len)
    
    '''
    
    attn_logits=tf.matmul(q,k,transpose_b=True)  #shape :(batch_size,num_attn_head,seq_len_q,seq_len_k)
    attn_logits_scaled=tf.divide(attn_logits,tf.sqrt(tf.cast(tf.shape(k)[-1],tf.float32))) #dividing by the square root of embedding dimension to set variance of attn_logits to 1 and avoiding pushing the softmax scores towards 0 or 1 (hard softmax)

    if mask is not None:
        attn_logits_scaled+=(mask*-1e12) #making the mask 

    attn_scaled=tf.nn.softmax(attn_logits_scaled,axis=-1) #softmax would be done across the keys dimension for getting the attention score of keys wrt to a query
    
    #shape(attn_scaled) :(batch_size,num_attn_head,seq_len_q,seq_len_k)
    
    out=tf.matmul(attn_scaled,v)
    
    return attn_scaled,out

class MultiHeadAttn(tf.keras.layers.Layer):
    
    def __init__(self,embed_dim,num_attn_head):
        super(MultiHeadAttn,self).__init__()
        
        '''
        
        One thing to note is that dimensionality % num of attention heads should be equal to zero because the idea is to 
        learn joint distribution from different linear projections of key,query and value matrix.
        
        params:
        dimension:the original dimension of initialized word embeddings
        attention_heads:how many different linear projections of key,query and value we want.
        
        '''
        
        self.embed_dim=embed_dim
        self.attn_head=num_attn_head
        
        assert self.embed_dim %self.attn_head ==0     #checking the above criteria
        
        self.depth =self.embed_dim//self.attn_head    #this would be the embedding dimension of k,q and v matrix after splitting in heads
       
        self.wq=tf.keras.layers.Dense(embed_dim)      #this layer will be used to linear project q matrix
        self.wv=tf.keras.layers.Dense(embed_dim)      #this layer will be used to linear project v matrix
        self.wk=tf.keras.layers.Dense(embed_dim)      #this layer will be used to linear project k matrix
        self.linear=tf.keras.layers.Dense(embed_dim)  #this layer will be used for linear projection after multi head split concatenation

    def __split__(self,batch_size,x):
        
        '''
        Use of this method is to split the incoming key,query,value matrix into a shape of {batch_size,self.attn_head,seq_length,self.depth} 
        This will help to learn the joint distribution of these matrices at different position wrt to different space.
        
        params:
        x:incoming k,q or v matrix
        batch_size:batch_size (number of queries)
        
        '''
        
        return tf.transpose(tf.reshape(x,(batch_size,-1,self.attn_head,self.depth)),perm=[0,2,1,3]) #the transpose is done to make shape of (batch_size,self.attn_head,seq_len,self.depth)

    
    def call(self,k,q,v,mask):
        
        batch_size=tf.shape(q)[0]
        
        #splitting the linearly transformed matrix to multiple small matrices.
        k=self.__split__(batch_size,self.wk(k))
        q=self.__split__(batch_size,self.wq(q))
        v=self.__split__(batch_size,self.wv(v))
        
        attn_scaled,out=scaled_dot_attention(k,q,v,mask) #shape(attn_scaled):(batch_size,num_attn_head,seq_len_q,seq_len_k)
                                                         #shape(out):(batch_size,num_attn_head,seq_len_q,self.depth)
            
        #restoring the output to shape:(batch_size,seq_len,self.embed_dim)
        
        out=tf.reshape(tf.transpose(out,perm=[0,2,1,3]),(batch_size,-1,self.embed_dim))
        
        #passing the output to the dense layer
        
        out=self.linear(out)
        
        return attn_scaled,out    

def ffn(higher_dim,embed_dim): #feed foward point wise network to pass the output obtained after multi head attn
    return tf.keras.Sequential([
                                tf.keras.layers.Dense(higher_dim,activation='relu'),
                                tf.keras.layers.Dense(embed_dim)
                               ])
    
class encoder_layer(tf.keras.layers.Layer):

    #Creates a single encoder layer which is (self-attention(multi-head)+ffn)
    
    def __init__(self,embed_dim,num_attn_head,higher_dim,drop_rate=0.05):
        super(encoder_layer,self).__init__()
        self.embed_dim=embed_dim
        self.attn_head=num_attn_head
        
        #initialising the multi head attention layer
        self.multi_attn=MultiHeadAttn(embed_dim,num_attn_head)
        
        #initialising the ffn network layer
        self.ffn=ffn(higher_dim,embed_dim)
        
        # dropout+layer norm
        self.dropout1=tf.keras.layers.Dropout(drop_rate,seed=seed_value)
        self.dropout2=tf.keras.layers.Dropout(drop_rate,seed=seed_value)
        
        '''
        LayerNormalization --> x=[x1,x2,x3,.......xn] where x is an embedding 
         
         1.calculate mean 
         2.calculate variance
         3.x1= alpha*(x1-mean)/sqrt(variance+epsilon)+beta where alpha and beta are learnable parameters.
         4.repeat this for all other xi.

        '''
        
        self.norm1=tf.keras.layers.LayerNormalization(epsilon=1e-9)
        self.norm2=tf.keras.layers.LayerNormalization(epsilon=1e-9)
        
        
    def call(self,x,mask,is_train):
        
        attn_scaled,out=self.multi_attn(x,x,x,mask) #out shape=(batch_size,seq_len,self.embed_dim)
        out=self.dropout1(out,training=is_train)    #out shape=(batch_size,seq_len,self.embed_dim)
        out=self.norm1(out+x)
        
        ffn_out=self.ffn(out)
        ffn_out=self.dropout2(ffn_out,training=is_train)
        ffn_out=self.norm2(ffn_out+out)
        
        return ffn_out                             #ffn_out shape=(batch_size,seq_len,self.embed_dim)




def angle_rate(pos,i,dim):
    return pos* 1/np.power(1e4,((2*(i/2))/dim))

def positional_embedding(max_sequence_length,embed_dim):

    #Used for positional encoding
    
    max_sequence=np.arange(max_sequence_length).reshape(max_sequence_length,1)
    dimension=np.arange(embed_dim).reshape(1,embed_dim)
    
    embeddings=angle_rate(max_sequence,dimension,embed_dim)
    embeddings[:,0::2]=np.sin(embeddings[:,0::2])
    embeddings[:,1::2]=np.cos(embeddings[:,1::2]) #shape=(max_sequence_length,embedding_dim)
    
    return tf.cast(embeddings[np.newaxis,:,:],tf.float32) #shape=(batch_size,max_sequence_length,embedding_dim)


class encoder_nx(tf.keras.layers.Layer):


    #Creates a stack of  encoders which consists  of self-attention(multi-head+ffn)
    
    def __init__(self,num_coders,num_attn_head,embed_dim,higher_dim,vocab_size,max_sequence_length,drop_rate=0.05):
        super(encoder_nx,self).__init__()
        self.num_encoders=num_coders
        self.embedding=tf.keras.layers.Embedding(vocab_size,embed_dim)  #init embedding matrix of shape (vocab_Size,embed_dim)
        self.dropout1=tf.keras.layers.Dropout(drop_rate,seed=seed_value)
        self.get_position_embed=positional_embedding(max_sequence_length,embed_dim)
        self.encoders=[encoder_layer(embed_dim,num_attn_head,higher_dim,drop_rate) for i in range(self.num_encoders)]
        
    
    def call(self,x,mask,is_train):
        
        sequence_length=tf.shape(x)[1]
        x_embedding=self.embedding(x) #get embedding from embedding layer
        x_embedding+=self.get_position_embed[:,:sequence_length,:] #add position embedding
        x_embedding=self.dropout1(x_embedding,training=is_train)#dropout layer
        
        for i in self.encoders:
            x_embedding=i(x_embedding,mask,is_train)
            
        return x_embedding #this is the encoder output after n encoders       


            

class LearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    
    def __init__(self,embed_dim=768,warmup_steps=8000):
        super(LearningRate,self).__init__()
        self.dim=tf.cast(embed_dim,tf.float32)
        self.warmup_steps=warmup_steps
        
    def __call__(self,step):
        
        return tf.math.rsqrt(self.dim) * tf.math.minimum(tf.math.rsqrt(step),step*(self.warmup_steps**-1.5))






class BAttention(tf.keras.layers.Layer):

    #Performs bahadanau attention
    
    def __init__(self):
       super(BAttention, self).__init__()

    
    def build(self, input_shape):
       
        self.W = self.add_weight(name="call_1", shape=(input_shape.as_list()[-1], 1),initializer="normal")
        self.bias = self.add_weight(name="call_me_bias", shape=(2000, 1),initializer="zeros")
        super(BAttention, self).build(input_shape)


    def call(self, x,mask=None):

        et = tf.squeeze((tf.matmul(x, self.W)+self.bias[:tf.shape(x)[1],]),axis=-1)
        if mask is not None:
            et+=(tf.squeeze(mask[:,:,:,:],axis=[1,2])*(-1e12)) #masking
        et = tf.nn.softmax(et,axis=-1)
        et=tf.expand_dims(et, axis=-1)
        et = x* et
        return tf.reduce_sum(et, axis=1)  

    def compute_mask(self, input, input_mask=None):
        return None   # do not pass the mask to the next layers

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[-1])


    def get_config(self):
        return super(BAttention, self).get_config()
    

class BAttentionTop(tf.keras.layers.Layer):

    #Performs bahadanau attention and generates context vector ony over top 5 attention scores ,rest are ignored 
    
    def __init__(self):
       super(BAttentionTop, self).__init__()

    
    def build(self, input_shape):
       
        self.W = self.add_weight(name="call_1", shape=(input_shape.as_list()[-1], 1),initializer="normal")
        super(BAttentionTop, self).build(input_shape)


    def call(self, x,mask=None):

        et = tf.squeeze(tf.math.tanh(tf.matmul(x, self.W)),axis=-1) 
        if mask is not None:
            et+=(tf.squeeze(mask[:,:,:,:],axis=[1,2])*(-1e12)) #masking

        values,_=tf.math.top_k(et,k=5)
        values=tf.expand_dims(values[:,-1],axis=-1)
        et=tf.cast(tf.math.greater_equal(et,tf.expand_dims(values[:,-1],-1)),tf.float32)*et
        et = tf.nn.softmax(et,axis=-1)
        et=tf.expand_dims(et, axis=-1)
        et = x* et
        return tf.reduce_sum(et, axis=1)  

    def compute_mask(self, input, input_mask=None):
        return None   # do not pass the mask to the next layers

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[-1])


    def get_config(self):
        return super(BAttentionTop, self).get_config()


class HBAttention(tf.keras.layers.Layer):

    #Performs bahadanau attention
    
    def __init__(self):
       super(HBAttention, self).__init__()

    
    def build(self, input_shape):
       
        self.W = self.add_weight(name="call_1", shape=(input_shape.as_list()[-1], 1),initializer="normal")
        super(HBAttention, self).build(input_shape)


    def call(self, x,mask=None):

        et = tf.squeeze((tf.matmul(x, self.W)),axis=-1)
        if mask is not None:
            et+=mask*(-1e12) #masking
        et = tf.nn.softmax(et,axis=-1)
        et=tf.expand_dims(et, axis=-1)
        et = x* et
        return tf.reduce_sum(et, axis=1)  

    def compute_mask(self, input, input_mask=None):
        return None   # do not pass the mask to the next layers

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[-1])


    def get_config(self):
        return super(HBAttention, self).get_config()
    
