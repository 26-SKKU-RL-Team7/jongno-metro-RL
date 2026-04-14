import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from src.jongno_env import JongnoMetroEnv

def main():
    print("🛡️ [FULL TRAIN] 정규 학습 환경을 초기화합니다...")
    
    # 평가(Evaluation)를 위해 별도의 독립된 환경을 만듭니다.
    env = Monitor(JongnoMetroEnv(max_budget=15))
    eval_env = Monitor(JongnoMetroEnv(max_budget=15))
    
    # 평가 콜백 (만 스텝마다 테스트하여 최고 기록 갱신 시 자동 저장)
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='./models/best_jongno_model/',
        log_path='./tensorboard_logs/results/', 
        eval_freq=10000, # 10,000 스텝마다 평가
        deterministic=True, 
        render=False
    )

    # 표준 DQN 하이퍼파라미터 (깊고 넓은 탐색)
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=1e-4,             # 세밀한 학습
        buffer_size=100000,             # 큰 메모리
        learning_starts=10000,          # 충분한 무작위 탐색(데이터 수집) 후 학습 시작
        batch_size=128,
        exploration_fraction=0.5,       # 전체 스텝의 50% 동안 서서히 탐색률 감소
        target_update_interval=1000,    # 타겟 네트워크 업데이트 주기
        policy_kwargs=dict(net_arch=[256, 256]), # 깊은 신경망
        tensorboard_log="./tensorboard_logs/"
    )
    
    print("🏋️‍♂️ 학습 시작 (총 500,000 Timesteps)...")
    model.learn(total_timesteps=500000, callback=eval_callback, tb_log_name="Full_DQN")
    
    model.save("models/final_jongno_dqn")
    print("✅ 최종 모델 저장 완료!")

if __name__ == "__main__":
    main()