import os
import sys
import time
import pygame
from stable_baselines3 import DQN

# 파이썬이 src 폴더 내부의 파일들을 직접 찾을 수 있도록 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.jongno_env import JongnoMetroEnv
from src.ui.viewport import Viewport
from config import screen_color

def main():
    pygame.init()
    
    # [수정] 기본 창 크기를 1280x720으로 설정 (Resizable 지원)
    display_width, display_height = 1280, 720
    window_surface = pygame.display.set_mode((display_width, display_height), pygame.RESIZABLE)
    pygame.display.set_caption("Jongno Metro DQN Simulation - Optimized View")
    
    # 환경 설정 (최대 예산 15)
    env = JongnoMetroEnv(max_budget=15)
    
    # 정규 학습 모델 로드 (경로 확인 필수)
    model_path = "models/jongno_dqn_model.zip"
    if not os.path.exists(model_path):
        # 만약 모델 이름이 다르다면 실제 저장된 파일명으로 수정하세요.
        model_path = "models/final_jongno_dqn.zip"
        
    try:
        model = DQN.load(model_path)
        print(f"성공적으로 모델({model_path})을 불러왔습니다.")
    except FileNotFoundError:
        print("저장된 모델이 없습니다. train_agent.py를 먼저 실행하여 학습을 완료하세요.")
        return

    obs, _ = env.reset()
    viewport = Viewport(env.mediator)
    clock = pygame.time.Clock()
    
    print("📺 시뮬레이션 시작 (10 FPS - 관찰 최적화 모드)")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # 창 크기가 변경되어도 Viewport 클래스 내의 draw 로직이 대응함
                
        # 1. 모델 추론 (결정론적 모드)
        action, _states = model.predict(obs, deterministic=True)
        
        # 2. 스텝 진행 (1초 단위 시뮬레이션)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 3. 화면 렌더링
        # Viewport가 1920x1080 가상 화면을 현재 창 크기에 맞춰 중앙 정렬함
        window_surface.fill(screen_color)
        viewport.draw(window_surface, env.mediator.time_ms)
        pygame.display.flip()
        
        # [수정] 실행 속도를 초당 10프레임으로 제한하여 열차의 움직임과 에이전트의 선택을 명확히 관찰
        clock.tick(5) 
        
        if terminated:
            print(f"🚩 [게임 종료] 최종 스코어: {env.mediator.score}")
            # [수정] 종료 직후 바로 리셋되지 않도록 2초간 화면 유지
            time.sleep(2)
            obs, _ = env.reset()
            viewport = Viewport(env.mediator)
            print("🔄 새로운 에피소드를 시작합니다...")

    pygame.quit()

if __name__ == "__main__":
    main()