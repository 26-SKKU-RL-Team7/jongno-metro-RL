import os
import sys
import csv
import numpy as np

# 파이썬이 src 폴더 내부를 찾을 수 있도록 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from src.jongno_env import JongnoMetroEnv

class JongnoLogger(BaseCallback):
    def __init__(self, filename="smoke_results.csv"):
        super().__init__()
        self.filename = filename
        # CSV 헤더 작성
        with open(self.filename, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(["Step", "Hour", "Action", "Reward", "Budget"])

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        obs = self.locals['new_obs'][0]
        action = self.locals['actions'][0]
        reward = self.locals['rewards'][0]
        # obs의 마지막 값이 시간 인덱스 (05시 기준 보정)
        hour = int(obs[-1]) + 5 
        
        with open(self.filename, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([self.num_timesteps, f"{hour}시", action, round(reward, 2), info.get('budget_left')])
        return True

def main():
    # 1. 모델 저장 폴더 생성 확인
    os.makedirs("models", exist_ok=True)

    print("🚀 [SMOKE TEST] 초고속 훈련 및 파일 저장 세팅을 시작합니다...")
    env = JongnoMetroEnv()
    
    # 2. 모델 정의 (빠른 결과를 위해 경량화 설정)
    model = DQN(
        "MlpPolicy", 
        env, 
        learning_starts=10, 
        batch_size=16, 
        policy_kwargs=dict(net_arch=[32, 32]),
        tensorboard_log="./tensorboard_logs/"
    )
    
    # 3. 학습 진행 (1000 스텝)
    print("🏃‍♂️ 학습 중...")
    model.learn(total_timesteps=1000, callback=JongnoLogger())
    
    # 4. 모델 저장 (시뮬레이션 코드와 이름 일치: smoke_jongno_dqn)
    model.save("models/smoke_jongno_dqn")
    
    print("✅ 학습 완료!")
    print("📍 저장된 모델: models/smoke_jongno_dqn.zip")
    print("📍 로그 파일: smoke_results.csv")

if __name__ == "__main__":
    main()