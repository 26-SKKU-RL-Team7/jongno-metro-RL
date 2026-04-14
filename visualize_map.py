import json
import matplotlib.pyplot as plt

# 한글 폰트 깨짐 방지 설정 (Windows: 맑은 고딕, Mac: AppleGothic)
import platform
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def visualize_jongno_map(config_path="jongno_config.json"):
    # 1. 전처리된 JSON 파일 로드
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"오류: '{config_path}' 파일이 없습니다. preprocess_jongno.py를 먼저 실행하세요.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # 호선별 대표 색상 지정 (실제 지하철 노선색과 유사하게)
    line_colors = {
        "1호선": "#0052A4", # 남색
        "3호선": "#EF7C1C", # 주황색
        "4호선": "#00A5DE", # 하늘색
        "5호선": "#996CAC", # 보라색
        "6호선": "#CD7C2F"  # 황토색
    }

    # 2. 노선(Edge) 그리기
    for line_name, stations in data["lines"].items():
        x_coords = []
        y_coords = []
        for st in stations:
            if st in data["stations"]:
                x_coords.append(data["stations"][st]["x"])
                y_coords.append(data["stations"][st]["y"])
        
        color = line_colors.get(line_name, "gray")
        # 노선 선 그리기 및 정점(역) 마커 표시
        ax.plot(x_coords, y_coords, marker='o', linewidth=4, markersize=10, 
                label=line_name, color=color, alpha=0.8, zorder=1)

    # 3. 역 이름(Node Label) 표시
    for st_name, coords in data["stations"].items():
        x, y = coords["x"], coords["y"]
        # 텍스트가 마커와 겹치지 않도록 약간 띄워서(offset) 출력
        ax.annotate(st_name, (x, y), xytext=(8, 8), textcoords='offset points', 
                    fontsize=11, fontweight='bold', zorder=2,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

    # 4. 화면 설정
    # Pygame은 좌상단이 (0,0)이고 아래로 갈수록 y가 증가하므로 y축을 뒤집어줌
    ax.invert_yaxis() 
    
    ax.set_title("종로구 지하철 노선망 시각화 (Pygame 렌더링 좌표 기준)", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("화면 X 좌표 (Pixel)")
    ax.set_ylabel("화면 Y 좌표 (Pixel)")
    
    # 범례 및 그리드 표시
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 여백 최적화 후 출력
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_jongno_map()