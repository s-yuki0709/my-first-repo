import gym
import numpy as np


env = gym.make('CartPole-v1')   # 環境の初期化
# env.reset()                     # 環境のリセット

# for i in range(200):
#     action = env.action_space.sample()  # ランダムに行動選択
#     observation, reward, done, info = env.step(action)
    
#     print("Step {}".format(i+1))
#     print("状態: {}".format(observation))
#     print("終了判定: {}".format(done))
    
#     env.render()
    
    
# 価値Q(s,a)を記録するQテーブルの作成
q_table = {}        # Qテーブル

# Q値の設定
def setQ(state, action, value):
    q_table[(state, action)] = value
    
# Q値の取得
def getQ(state, action):
    
    # テーブルに状態が存在しないとき
    if not(state, action) in q_table:
        q_table[(state, action)] = 0
        
    return q_table[(state, action)]

# 状態の離散化
'''
CartPoleでは、カートの位置、カートの速度、ポールの位置、ポール速度を連続値で取得する
ただこの状態ではQテーブルに記録できないため、離散値に変換する必要がある
例）カートの位置
---------------------
|    範囲    |離散値|
|-2.4 ~ -1.6 |   1  |
|-1.6 ~ -0.8 |   2  | 
|-0.8 ~  0   |   3  |
| 0   ~  0.8 |   4  |
| 0.8 ~  1.6 |   5  |
| 1.6 ~  2.4 |   6  |
---------------------
'''


BIN_NUMBER = 6  # 離散値の数

# 離散値の範囲
bins = []
bins.append(np.linspace(-2.4, 2.4, BIN_NUMBER))     # カートの位置
bins.append(np.linspace(-3.0, 3.0, BIN_NUMBER))     # カートの速度
bins.append(np.linspace(-0.2, 0.2, BIN_NUMBER))     # ポールの角度
bins.append(np.linspace(-2.0, 2.0, BIN_NUMBER))     # ポールの速度

# 観測データを状態（離散値）に変換
def digitize(observation):
    
    state = []
    
    state.append(np.digitize(observation[0], bins[0]))
    state.append(np.digitize(observation[1], bins[1]))
    state.append(np.digitize(observation[2], bins[2]))
    state.append(np.digitize(observation[3], bins[3]))
    
    return tuple(state)

# 
# env.reset()     # 環境のリセット

# for i in range(3):
#     action = env.action_space.sample()
#     observation, reward, done, info = env.step(action)
    
#     print(observation)
#     print(digitize(observation))
    
    
# 報酬の取得
def getReward(step, done):
    
    if done:
        if step >= 180:
            reward = 1  # 目標ステップに到達
        else:
            reward = -200   # ペナルティ
    else:
        reward = 1
        
    return reward


alpha = 0.1     # 学習率
gamma = 0.9     # 割引率

# Q値の更新
def updataQTable(state, action, next_state, reward):
    
    max_value = max(getQ(next_state, 0), getQ(next_state, 1))
    value = (1 - alpha) * getQ(state, action) + alpha * (reward + gamma * max_value)
    
    setQ(state, action, value)
    
    
# env.reset()     # 環境のリセット
# q_table = {}    # Qテーブルの初期化

# # 最初の状態
# action = env.action_space.sample()
# observation, reward, done, info = env.step(action)
# state = digitize(observation)
# print("state: {}".format(state))
# print("action: {}".format(action))

# # 次の状態
# next_action = env.action_space.sample()
# observation, reward, done, info = env.step(action)
# next_state = digitize(observation)
# print("next_state: {}".format(next_state))

# # 報酬を取得
# reward = getReward(0, done)
# print("reward: {}".format(reward))

# # Qテーブルの更新
# updataQTable(state, action, next_state, reward)

# # Q値の確認
# q_value = q_table[(state, action)]
# print("Q value: {}".format(q_value))


# 行動の選択
# ε-greedy法を用いる
'''
ε-greedy法
確率εでランダムな行動を選択し、
確率1-εで状態sにおいて最も価値Q(s, a)が大きい行動aを選択する
'''
def greedyAction(state, epsilon):
    
    # np.random.rand()：一様分布（0.0以上、1.0未満）
    if epsilon > np.random.rand():
        action = env.action_space.sample()
    else:
        action = np.argmax([getQ(state, 0), getQ(state, 1)])
        
    return action


# Qテーブルの学習
'''
200ステップのエピソードを1000回繰り返してQテーブルを学習する
ε = 0.2に設定
'''
env.reset()     # 環境のリセット
q_table = {}    # Qテーブルの初期化

for episode in range(1000):
    print("エピソード[{}]".format(episode))
    observation = env.reset()
    
    for i in range(200):
        # 状態の取得
        state = digitize(observation)
        # ε-greedy法で行動選択
        action = greedyAction(state, 0.2)
        # 次の状態に遷移
        observation, reward, done, info = env.step(action)
        # 次の状態
        next_state = digitize(observation)
        # 報酬の取得
        reward = getReward((i+1), done)
        # Q値の更新
        updataQTable(state, action, next_state, reward)
        
        if done:
            break

# 学習したQテーブルを用いて実行        
env.reset()     # 環境のリセット
observation = env.reset()

for i in range(200):
    # 状態の取得
    state = digitize(observation)
    # ε-greedy法で行動選択
    action = greedyAction(state, 0)
    # 次の状態に遷移
    observation, reward, done, info = env.step(action)
    
    print("Step {}".format(i + 1))
    print("状態: {}".format(observation))
    print("終了判定: {}".format(done))
    
    env.render()    # 環境の描画