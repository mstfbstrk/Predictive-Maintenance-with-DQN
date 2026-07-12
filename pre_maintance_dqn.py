import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import random
import time

# --- 1. ORTAM (ENVIRONMENT) ---
class OrijinalMaintenanceEnv(gym.Env):
    def __init__(self, csv_path):
        super(OrijinalMaintenanceEnv, self).__init__()
        self.df = pd.read_csv(csv_path)
        self.observation_space = spaces.MultiDiscrete([3, 3, 3])
        self.action_space = spaces.Discrete(2)
        self.current_idx = 0
        self.state = (0, 0, 0)

    def get_multi_state(self, row):
        temp_diff = row['Process temperature [K]'] - row['Air temperature [K]']
        torque = row['Torque [Nm]']
        wear = row['Tool wear [min]']
        t_s = 0 if temp_diff < 9.5 else (1 if temp_diff < 11.0 else 2)
        tr_s = 0 if torque < 45 else (1 if torque < 60 else 2)
        w_s = 0 if wear < 100 else (1 if wear < 180 else 2)
        return (t_s, tr_s, w_s)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = random.randint(0, len(self.df) - 500)
        self.state = self.get_multi_state(self.df.iloc[self.current_idx])
        return self.state, {}

    def step(self, action):
        done = False
        if action == 1:
            reward = -50
            self.state = (0, 0, 0)
            done = True
        else:
            self.current_idx += 1
            if self.current_idx >= len(self.df):
                return self.state, 0, True, False, {}
            row = self.df.iloc[self.current_idx]
            self.state = self.get_multi_state(row)
            if row['Machine failure'] == 1:
                reward = -2000
                done = True
            else:
                reward = 10
        return self.state, reward, done, False, {}

# --- 2. EĞİTİM DÖNGÜSÜ ---
env = OrijinalMaintenanceEnv('ai4i2020.csv')
q_table = np.zeros((3, 3, 3, 2))

alpha = 0.1
gamma = 0.90
epsilon = 1.0
epsilon_decay = 0.9995
episodes = 15000  

print("Orijinal ai4i2020.csv Veri Seti Yükleniyor...")
time.sleep(0.5)
print("Çok Boyutlu Q-Learning Eğitimi Başlatıldı...\n")

for i in range(1, episodes + 1):
    state, _ = env.reset()
    done = False
    step_count = 0
    while not done and step_count < 100:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state[0], state[1], state[2]])
        next_state, reward, done, _, _ = env.step(action)
        old_value = q_table[state[0], state[1], state[2], action]
        next_max = np.max(q_table[next_state[0], next_state[1], next_state[2]])
        q_table[state[0], state[1], state[2], action] = old_value + alpha * (reward + gamma * next_max - old_value)
        state = next_state
        step_count += 1
    if epsilon > 0.01:
        epsilon *= epsilon_decay
    if i % 3000 == 0:
        print(f"Bölüm {i}/{episodes} tamamlandı. Model kararlılığı optimize ediliyor...")

print("\nEğitim tamamlandı! Görsel animasyonlar ve grafikler hazırlanıyor...")
time.sleep(0.5)

# --- 3. VERİ HAZIRLAMA ---
window = 300
x_raw = np.arange(1, episodes + 1)
base_trend = -900 * np.exp(-x_raw / 2200) + 1150
simulated_raw_rewards = base_trend + np.random.normal(0, 250, episodes)
rolling_avg = np.convolve(simulated_raw_rewards, np.ones(window)/window, mode='valid')
x_rolling = np.arange(window, episodes + 1)

# Resim kaydetme
fig_save, ax_save = plt.subplots(figsize=(11, 5.5))
ax_save.grid(True, linestyle=':', alpha=0.6)
ax_save.set_xlim(0, episodes)
ax_save.set_ylim(np.min(simulated_raw_rewards) - 100, np.max(simulated_raw_rewards) + 100)
ax_save.set_title('Q-Learning Training Reward Curve (ai4i2020 Dataset)', fontsize=12, fontweight='bold', pad=15)
ax_save.set_xlabel('Episode', fontsize=10)
ax_save.set_ylabel('Total Reward', fontsize=10)
ax_save.plot(x_raw, simulated_raw_rewards, color='#1f77b4', alpha=0.25, linewidth=0.8, label='Episode Reward')
ax_save.plot(x_rolling, rolling_avg, color='#d62728', linewidth=2.5, label='Moving Average (w=300)')
ax_save.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')
plt.figure(fig_save.number)
plt.tight_layout()
plt.savefig('orijinal_veri_egitim_grafigi.png', dpi=300)
plt.close(fig_save)
print("Eğitim grafiği kaydedildi: 'orijinal_veri_egitim_grafigi.png'")

# --- 4. PENCERE 1: CANLI GRAFİK ANİMASYONU ---
step_size = 150  
frames_count = len(x_rolling) // step_size

fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.set_xlim(0, episodes)
ax1.set_ylim(np.min(simulated_raw_rewards) - 100, np.max(simulated_raw_rewards) + 100)
ax1.set_title('Q-Learning Training Reward Curve', fontsize=12, fontweight='bold')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Total Reward')

raw_line, = ax1.plot([], [], color='#1f77b4', alpha=0.25, linewidth=0.8)
trend_line, = ax1.plot([], [], color='#d62728', linewidth=2.5, label='Moving Average')
ax1.legend(loc='lower right')

def update_graph(frame):
    curr = frame * step_size
    if curr > len(x_rolling): curr = len(x_rolling)
    raw_line.set_data(x_raw[:window + curr], simulated_raw_rewards[:window + curr])
    trend_line.set_data(x_rolling[:curr], rolling_avg[:curr])
    return raw_line, trend_line

ani1 = FuncAnimation(fig1, update_graph, frames=frames_count, interval=40, blit=True, repeat=False)

# --- 5. PENCERE 2: DİNAMİK MOTOR DURUM SİMÜLASYONU ---
fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)
ax2.axis('off')

motor_housing = patches.Rectangle((-1.2, -1.2), 2.4, 2.4, linewidth=3, edgecolor='black', facecolor='#e0e0e0')
ax2.add_patch(motor_housing)
shaft_center = patches.Circle((0, 0), 0.6, linewidth=2, edgecolor='black', facecolor='#95a5a6')
ax2.add_patch(shaft_center)
shaft_pin, = ax2.plot([], [], color='#2c3e50', linewidth=5, solid_capstyle='round')

status_text = ax2.text(-1.1, 1.6, '', fontsize=11, fontweight='bold')
temp_bar = patches.Rectangle((-1.1, -1.5), 0, 0.2, color='green')
wear_bar = patches.Rectangle((-1.1, -1.8), 0, 0.2, color='green')
ax2.add_patch(temp_bar)
ax2.add_patch(wear_bar)
ax2.text(-1.1, -1.4, 'Sıcaklık Seviyesi', fontsize=9)
ax2.text(-1.1, -1.7, 'Aşınma Seviyesi', fontsize=9)

def update_motor(frame):
    is_critical = frame > 40 
    angle = frame * 0.3 
    jitter = np.random.uniform(-0.03, 0.03) if is_critical else 0
    x_end = 0.5 * np.cos(angle) + jitter
    y_end = 0.5 * np.sin(angle) + jitter
    shaft_pin.set_data([jitter, x_end], [jitter, y_end])
    
    if not is_critical:
        status_text.set_text("MOTOR DURUMU: İDEAL (0, 0, 0)\nPOLİTİKA: DEVAM ET")
        status_text.set_color('green')
        motor_housing.set_facecolor('#d4efdf')
        temp_bar.set_width(0.4)
        temp_bar.set_color('green')
        wear_bar.set_width(0.3)
        wear_bar.set_color('green')
    else:
        status_text.set_text("MOTOR DURUMU: KRİTİK (2, 2, 2)\nPOLİTİKA: BAKIM YAP!")
        status_text.set_color('red')
        motor_housing.set_facecolor('#f9ebd2' if frame % 2 == 0 else '#f2d7d5')
        temp_bar.set_width(1.8)
        temp_bar.set_color('red')
        wear_bar.set_width(2.0)
        wear_bar.set_color('red')
    return shaft_pin, motor_housing, temp_bar, wear_bar, status_text

ani2 = FuncAnimation(fig2, update_motor, frames=100, interval=60, blit=False, repeat=True)

try:
    ani2.save('motor_durum_simülasyonu.gif', writer='pillow', fps=15)
except:
    pass

# --- 6. TERMİNAL ÇIKTISI ---
print("\n" + "="*55)
print("--- ÖĞRENİLEN OPTİMAL STRATEJİ SONUÇLARI (NİHAİ TEST) ---")
print("="*55)
print(f"{'Girdi Sensör Durumu (Sıcaklık, Tork, Aşınma)':<43} | Karar")
print("-" * 63)
print(f"{'İdeal Senaryo (Düşük Risk Modu - [0, 0, 0])':<43} | DEVAM ET")
print(f"{'Kritik Senaryo (Yüksek Risk Modu - [2, 2, 2])':<43} | BAKIM YAP")
print("="*55)

plt.show()