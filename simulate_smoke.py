import os
import sys
import pygame
import time
from stable_baselines3 import DQN

# 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.jongno_env import JongnoMetroEnv
from src.ui.viewport import Viewport
# config에서 해상도를 가져오되, 시뮬레이션 창은 작게 띄우기 위해 변수 조정
from config import screen_color

def main():
    pygame.init()
    
    # [수정] 전체 화면(1920x1080) 대신 적절한 창 크기로 설정
    display_width, display_height = 1280, 720
    window_surface = pygame.display.set_mode((display_width, display_height), pygame.RESIZABLE)
    pygame.display.set_caption("Jongno Metro - RL Agent Observation")
    
    env = JongnoMetroEnv(max_budget=15)
    
    model_path = "models/smoke_jongno_dqn.zip"
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일({model_path})을 찾을 수 없습니다. smoke_train_agent.py를 먼저 실행하세요.")
        return
        
    model = DQN.load(model_path)
    obs, _ = env.reset()
    viewport = Viewport(env.mediator)
    clock = pygame.time.Clock()
    
    print("📺 시뮬레이션 시작 (10 FPS - 저속 모드)")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # 1. 모델 추론
        action, _ = model.predict(obs, deterministic=True)
        
        # 2. 환경 업데이트 (1스텝당 1초 진행)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 3. 렌더링 (Viewport가 1920x1080을 현재 창 크기인 1280x720으로 자동 스케일링함)
        window_surface.fill(screen_color)
        viewport.draw(window_surface, env.mediator.time_ms)
        pygame.display.flip()
        
        # [수정] 속도 조절: 초당 10번만 업데이트 (게임 시간 10초/실제 1초)
        # 너무 빠르면 5 정도로 더 낮추셔도 됩니다.
        clock.tick(10) 
        
        if terminated:
            print(f"🚩 [게임 종료] 최종 점수: {env.mediator.score}")
            # 즉시 리셋되지 않도록 2초간 멈춤 (상황 파악용)
            time.sleep(2)
            obs, _ = env.reset()
            viewport = Viewport(env.mediator)
            print("🔄 환경을 리셋하고 다시 시작합니다...")

    pygame.quit()

if __name__ == "__main__":
    main()