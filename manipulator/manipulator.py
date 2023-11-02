import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class ManipulatorEnvironment:
    def __init__(self):
        self.upper_len = 12
        self.lower_len = 9
        self.wrist_len = 3
        self.state = np.array([90, -150, 40, 7.319078, 3.179711]) #[servo1, servo2, servo3, end-eff_x, end-eff_y]
        
        self.observation_space = 5 # servo1 angle, servo2 angle, servo3 angle, end-effector position
        self.action_space = 6
        self.target_x = 15
        self.target_y = 8

        self.prev_delta = np.sqrt((self.state[3] - self.target_x)**2+(self.state[4] - self.target_y)**2)

        # minimal and maximal value of each servo to create good behavior
        self.upper_min = 30
        self.upper_max = 120
        self.lower_min = -200
        self.lower_max = -50
        self.wrist_min = -20
        self.wrist_max = 70

        # for animation
        self.upper_x = [[0], [self.upper_len]]
        self.lower_x = [[self.upper_x[1][0]], [self.upper_x[1][0] + self.lower_len]]
        self.wrist_x = [[self.lower_x[1][0]], [self.lower_x[1][0] + self.wrist_len]]

        self.upper_y = [[0], [0]]
        self.lower_y = [[0], [0]]
        self.wrist_y = [[0], [0]]

        self.seq_upper_y = [[0], [0]]
        self.seq_lower_y = [[0], [0]]
        self.seq_wrist_y = [[0], [0]]

        self.seq_upper_x = [[0], [self.upper_len]]
        self.seq_lower_x = [[self.upper_x[1][0]], [self.upper_x[1][0] + self.lower_len]]
        self.seq_wrist_x = [[self.lower_x[1][0]], [self.lower_x[1][0] + self.wrist_len]]

        self.rot_angle1 = [0, 90]
        self.rot_angle2 = [0, -150]
        self.rot_angle3 = [0, 40]

        self.fig_anim, self.ax_anim = plt.subplots()
        self.upper, = self.ax_anim.plot([], [], 'black', linewidth=7.0)
        self.lower, = self.ax_anim.plot([], [], 'black', linewidth=7.0)
        self.wrist, = self.ax_anim.plot([], [], 'black', linewidth=7.0)
        self.target, = self.ax_anim.plot([], [], 'r', linewidth=7.0)

    def reset(self):
        self.state = np.array([90, -150, 40, 7.319078, 3.179711]) #[servo1, servo2, servo3, end-eff_x, end-eff_y]
        return self.state

    def step(self, action):
        # Simulate the environment dynamics and generate a new state.
        reward = 0
        done = False
        if(action==0):
            self.state[0] += 1
        elif(action==1):
            self.state[0] -= 1
        elif(action==2):
            self.state[1] += 1
        elif(action==3):
            self.state[1] -= 1
        elif(action==4):
            self.state[2] += 1
        elif(action==5):
            self.state[2] -= 1
        
        self.forwardKinematics(self.state[0], self.state[1], self.state[2])

        new_state = np.array([self.state[0], self.state[1], self.state[2], self.state[3], self.state[4]])

        # Calculate the reward.
        # easy task example
        # if(self.state[3] >= self.target_x)\
        # and (self.state[4] >= self.target_y):
        #     reward = 0
        #     done = True
        if(self.state[3] >= self.target_x - 0.7 and self.state[3] <= self.target_x + 0.7)\
        and (self.state[4] >= self.target_y - 1 and self.state[4] <= self.target_y + 1):
            reward = 0
            done = True
        else:
            delta = np.sqrt((self.state[3] - self.target_x)**2+(self.state[4] - self.target_y)**2)
            reward = -delta*100

            if(self.state[0] > self.upper_max or self.state[0] < self.upper_min):
                if(self.state[0] > self.upper_max):
                    reward = -100000
                else:
                    reward = -100000
                done = True
            elif(self.state[1] > self.lower_max or self.state[1] < self.lower_min):
                if(self.state[1] > self.lower_max):
                    reward = -100000
                else:
                    reward = -100000
                done = True
            elif(self.state[2] > self.wrist_max or self.state[2] < self.wrist_min):
                if(self.state[2] > self.wrist_max):
                    reward = -100000
                else:
                    reward = -100000
                done = True

            if(self.state[4] < 0):
                reward = -100000
                done = True
            
            self.prev_delta = delta

        return new_state, reward, done, {}
    
    def forwardKinematics(self, angle1, angle2, angle3):

        self.state[3] = self.upper_len*np.cos(np.deg2rad(angle1)) \
            + self.lower_len*np.cos(np.deg2rad(angle1+angle2)) \
                + self.wrist_len*np.cos(np.deg2rad(angle1+angle2+angle3))
        
        self.state[4] = self.upper_len*np.sin(np.deg2rad(angle1)) \
            + self.lower_len*np.sin(np.deg2rad(angle1+angle2)) \
                + self.wrist_len*np.sin(np.deg2rad(angle1+angle2+angle3))
    
    def generate_rot_angle(self, policy, epsilon=0.2):
        state = self.reset()
        state = torch.from_numpy(state).unsqueeze(dim=0).float()
        done = False
        
        while not done:
            p = policy(state)
            p = p.item()
            if isinstance(p, np.ndarray):
                action = np.random.choice(4, p=p)
            else:
                action = p
            next_state, reward, done, _ = self.step(action)

            next_state = torch.from_numpy(next_state).unsqueeze(dim=0).float()
            state = state.squeeze(0)
            temp1 = float(state[0])
            temp2 = float(state[1])
            temp3 = float(state[2])
            # print(state)

            state = next_state.squeeze(0)
            state_1 = float(state[0])
            state_2 = float(state[1])
            state_3 = float(state[2])
            
            self.rot_angle1.append(-1*(temp1 - state[0]))
            self.rot_angle2.append(-1*(temp2 - state[1]))
            self.rot_angle3.append(-1*(temp3 - state[2]))

            state = state.unsqueeze(dim=0).float()
        
        print(f"Final Reward: {reward}")
        print(f"Servo  1: {self.state[0]}")
        print(f"Servo  2: {self.state[1]}")
        print(f"Servo  3: {self.state[2]}")
        print(f"End Effector: {self.state[3]}, {self.state[4]}")
    
    def initAnimation(self):
        self.ax_anim.set_xlim(-2, 30)
        self.ax_anim.set_ylim(-2, 30)
        return self.upper,self.lower,self.wrist,
    
    def updateAnimation(self, frame):
        self.upper.set_data([self.upper_x[0][int(frame)], self.upper_x[1][int(frame)]],\
                    [self.upper_y[0][int(frame)], self.upper_y[1][int(frame)]])
        
        self.lower.set_data([self.lower_x[0][int(frame)], self.lower_x[1][int(frame)]],\
                    [self.lower_y[0][int(frame)], self.lower_y[1][int(frame)]])
        
        self.wrist.set_data([self.wrist_x[0][int(frame)], self.wrist_x[1][int(frame)]],\
                    [self.wrist_y[0][int(frame)], self.wrist_y[1][int(frame)]])
        
        self.target.set_data([self.target_x, self.target_x], [self.target_y-1, self.target_y+1])

        return self.upper, self.lower, self.wrist, self.target,

    def test_agent(self, policy):
        temp1 = 0
        temp2 = 0
        temp3 = 0

        self.generate_rot_angle(policy)
        # print(self.rot_angle1)
        for j in range(1, len(self.rot_angle1)):
            t1 = np.linspace(temp1, temp1 + self.rot_angle1[j], 40)
            t2 = np.linspace(temp2, temp2 + self.rot_angle2[j], 40)
            t3 = np.linspace(temp3, temp3 + self.rot_angle3[j], 40)

            temp1+=self.rot_angle1[j]
            temp2+=self.rot_angle2[j]
            temp3+=self.rot_angle3[j]

            # link 1
            for i in range(0, 40):
                # x axis
                self.upper_x[0].append(0)
                self.upper_x[1].append(0 + self.upper_len*np.cos(np.deg2rad(t1[i])))   

                # y axis
                self.upper_y[0].append(0)
                self.upper_y[1].append(0 + self.upper_len*np.sin(np.deg2rad(t1[i])))

            # print("TOL",len(upper_x[0]))
            for i in range(0, 40):
                self.lower_x[0].append(self.upper_x[1][int(len(self.lower_x[0]))])
                self.lower_x[1].append(self.upper_x[1][int(len(self.lower_x[1]))] + self.lower_len*np.cos(np.deg2rad(t1[i] + t2[i])))

                self.lower_y[0].append(self.upper_y[1][int(len(self.lower_y[0]))])
                self.lower_y[1].append(self.upper_y[1][int(len(self.lower_y[1]))] + self.lower_len*np.sin(np.deg2rad(t1[i] + t2[i])))

            for i in range(0, 40):
                self.wrist_x[0].append(self.lower_x[1][int(len(self.wrist_x[0]))])
                self.wrist_x[1].append(self.lower_x[1][int(len(self.wrist_x[1]))] + self.wrist_len*np.cos(np.deg2rad(t1[i] + t2[i] + t3[i])))
                
                self.wrist_y[0].append(self.lower_y[1][int(len(self.wrist_y[0]))])
                self.wrist_y[1].append(self.lower_y[1][int(len(self.wrist_y[1]))] + self.wrist_len*np.sin(np.deg2rad(t1[i] + t2[i] + t3[i])))

            
            self.seq_upper_x[0].append(self.upper_x[0][len(self.upper_x[0])-1])
            self.seq_lower_x[0].append(self.lower_x[0][len(self.lower_x[0])-1])
            self.seq_wrist_x[0].append(self.wrist_x[0][len(self.wrist_x[0])-1])

            self.seq_upper_x[1].append(self.upper_x[1][len(self.upper_x[1])-1])
            self.seq_lower_x[1].append(self.lower_x[1][len(self.lower_x[1])-1])
            self.seq_wrist_x[1].append(self.wrist_x[1][len(self.wrist_x[1])-1])

            self.seq_upper_y[0].append(self.upper_y[0][len(self.upper_x[0])-1])
            self.seq_lower_y[0].append(self.lower_y[0][len(self.lower_x[0])-1])
            self.seq_wrist_y[0].append(self.wrist_y[0][len(self.wrist_x[0])-1])

            self.seq_upper_y[1].append(self.upper_y[1][len(self.upper_x[1])-1])
            self.seq_lower_y[1].append(self.lower_y[1][len(self.lower_x[1])-1])
            self.seq_wrist_y[1].append(self.wrist_y[1][len(self.wrist_x[1])-1])

        ani = FuncAnimation(self.fig_anim, self.updateAnimation, frames=np.linspace(0, int(len(self.upper_x[0])-1), int(len(self.upper_x[0])-1)), interval=1, repeat = False,
                    init_func=self.initAnimation, blit=True)
        # ani.pause()
        print(len(self.upper_x[0]))
        plt.show()
        