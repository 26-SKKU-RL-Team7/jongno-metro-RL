import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def plot_enhanced_results(log_path="smoke_results.csv"):
    # 1. 인코딩 에러 방지하며 읽기
    try:
        df = pd.read_csv(log_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(log_path, encoding='cp949')

    # 2. 데이터 전처리: '6시' -> 6 (숫자)로 변환
    df['Hour_Num'] = df['Hour'].str.replace('시', '').astype(int)
    
    # 그래프 영역 설정 (3단 구성)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    # [그래프 1] 누적 보상 추이 (성능 확인)
    df['Reward_MA'] = df['Reward'].rolling(window=20).mean()
    ax1.plot(df['Step'], df['Reward'], alpha=0.2, color='gray', label='Raw Reward')
    ax1.plot(df['Step'], df['Reward_MA'], color='blue', linewidth=2, label='이동 평균(20)')
    ax1.set_title("1. 학습 스텝별 보상(Reward) 추이", fontsize=14)
    ax1.set_ylabel("보상 점수")
    ax1.legend()

    # [그래프 2] 시간대별 평균 보상 (시간대별 난이도/성능)
    hourly_avg_reward = df.groupby('Hour_Num')['Reward'].mean()
    hourly_avg_reward.plot(kind='bar', ax=ax2, color='orange', edgecolor='black')
    ax2.set_title("2. 시간대별 평균 보상 (어느 시간대에 잘 대처했나?)", fontsize=14)
    ax2.set_ylabel("평균 보상")
    ax2.set_xlabel("시간 (Hour)")

    # [그래프 3] 시간대별 액션 히트맵 (에이전트의 선택 분석)
    # 시간대(Row)별로 어떤 Action(Col)을 많이 했는지 빈도 계산
    action_heatmap = pd.crosstab(df['Hour_Num'], df['Action'])
    sns.heatmap(action_heatmap, annot=True, fmt="d", cmap="YlGnBu", ax=ax3)
    ax3.set_title("3. 시간대별 액션 선택 빈도 (배차 전략 분석)", fontsize=14)
    ax3.set_xlabel("Action (0:대기, 1~5:추가, 6~10:회수)")
    ax3.set_ylabel("시간 (Hour)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_enhanced_results()