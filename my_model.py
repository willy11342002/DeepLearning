import tensorflow as tf

def layer(inputs, weight_shape, bias_shape):
    '''定義一個 relu 層'''
    
    # 設定初始化方法, 由於 relu 的特性, 所以使用常態分配來初始化
    w_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
    bias_init = tf.constant_initializer(value=0)
    
    # 定義每一層有 bias 個神經元
    w = tf.get_variable('w', weight_shape, initializer=w_init)
    b = tf.get_variable('b', bias_shape, initializer=bias_init)
    
    # 回傳使用 relu 運算
    return tf.nn.relu( tf.matmul(inputs, w) + b )

def inference(x):
    '''定義推測的步驟, 傳入一筆或多筆資料, 經過兩個隱藏層後, 直接輸出推測結果'''
    
    # 定義兩個隱藏層, 兩個隱藏層都有256個神經元, 與上一層全連結
    # 特別使用 variable_scope 這個特殊的方法, 這在圖上可以清楚看到
    # 後面會看到, 當我們要取得裡面的節點, 就需要加上這個變數區域
    with tf.variable_scope('hidden_1'):
        hidden_1 = layer(x, [784, 256], [256])
    with tf.variable_scope('hidden_2'):
        hidden_2 = layer(hidden_1, [256, 256], [256])
        
    # 定義一個輸出層, 與上一層全連結, 特別注意沒有softmax運算
    with tf.variable_scope('output'):
        output = layer(hidden_2, [256, 10], [10])
        
    return hidden_1, hidden_2, output
    
def loss(output, y):
    '''定義誤差函數計算的步驟, 這邊使用的是交叉嫡'''
    
    # 這個模型在這時候才使用 softmax 並進行交叉嫡運算
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y)    
    
    # 直接將所有的數字揉在一起做平均
    loss = tf.reduce_mean(xentropy)
    
    return loss

def training(cost, global_step, lr):   
    '''定義訓練的步驟, 用梯度下降法'''
    
    # 紀錄過程
    tf.summary.scalar('cost', cost)
    
    # 定義訓練的方法, 使用梯度下降法
    optimizer = tf.train.GradientDescentOptimizer(lr)
    
    # 進行誤差最小化任務
    train_op = optimizer.minimize(cost, global_step=global_step)
    
    return train_op
    
def training_no_optimizer(cost, optimizer, global_step, name='train'):   
    '''定義訓練的步驟, 用梯度下降法'''
    
    # 紀錄過程
    tf.summary.scalar('cost', cost)
       
    # 進行誤差最小化任務
    train_op = optimizer.minimize(cost, global_step=global_step, name='train')
    
    return train_op
    
def evaluate(output, y, name='eva'):
    '''定義評估的方式, 輸入標籤以及預測標籤, 輸出準確率'''
    
    # 找出標籤與預測標籤的最大信心水準, 比較是否相同
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    
    # 沿著 0 維度降維, 算出一個準確率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='eva')
    
    # 紀錄過程
    tf.summary.scalar('validation_error', (1. - accuracy))
    
    
    return accuracy
    
