import gym
import numpy as np

env = gym.make('CartPole-v1')   # 環境の初期化
print("Action Space: {}".format(env.action_space))  # 行動空間
print("Env Space {}".format(env.observation_space)) # 状態空間

observation = env.reset()   # 環境の初期化

cart_position = observation[0]      # カートの位置
cart_speed = observation[1]         # カートの速度
pole_angle = observation[2]         # ポールの角度
pole_speed = observation[3]         # ポールの速度

# print("カートの位置: {}".format(cart_position))
# print("カートの速度: {}".format(cart_speed))
# print("ポールの角度: {}".format(pole_angle))
# print("ポールの速度: {}".format(pole_speed))

action = 0  # 左に向かって力を加える

for i in range(10):
    env.step(action)
    
    observation, reward, done, info = env.step(action)

    print("状態: {}".format(observation))
    print("終了判定: {}".format(done))
    
env.render()    # 環境の描画