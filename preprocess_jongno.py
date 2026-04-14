import pandas as pd
import numpy as np
import json
import os

# 1. 설정 및 타겟 역 정의
JONGNO_STATIONS = [
    "종각", "종로3가", "종로5가", "동대문", "동묘앞", 
    "혜화", "안국", "경복궁", "광화문", "서대문", "독립문", "창신"
]

# 화면 해상도 및 안전 여백(Margin) 설정
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
MARGIN = 100  # 가장자리에서 최소 100픽셀 안쪽으로 역을 배치

def report_missing(found_list, target_list, file_name):
    found_set = set(found_list)
    target_set = set(target_list)
    missing = target_set - found_set
    if missing:
        print(f"⚠️  [데이터 누락] {file_name} 파일에 다음 역 정보가 없습니다: {missing}")
    else:
        print(f"✅  {file_name}: 모든 타겟 역이 로드되었습니다.")

def preprocess_all_data():
    config_data = {"stations": {}, "lines": {}, "demand": {}}
    
    # ---------------------------------------------------------
    # 1. 역사 좌표 데이터 처리 (여백 보정 정규화)
    # ---------------------------------------------------------
    print("\n--- [Step 1] 좌표 데이터 처리 및 여백 보정 정규화 ---")
    try:
        df_coords = pd.read_excel("전체_도시철도역사정보_20250930.xlsx", engine='openpyxl')
    except:
        df_coords = pd.read_csv("전체_도시철도역사정보_20250930.xlsx - Sheet1.csv", encoding='cp949')

    valid_coords = []
    found_names = []

    for target in JONGNO_STATIONS:
        # 부분 일치 검색으로 '경복궁(정부서울청사)' 등 명칭 이슈 해결
        match = df_coords[df_coords['역사명'].str.contains(target, na=False)].head(1)
        if not match.empty:
            valid_coords.append({
                "name": target,
                "lat": match.iloc[0]['역위도'],
                "lon": match.iloc[0]['역경도']
            })
            found_names.append(target)
    
    report_missing(found_names, JONGNO_STATIONS, "좌표 데이터(xlsx)")

    if valid_coords:
        lats = [v['lat'] for v in valid_coords]
        lons = [v['lon'] for v in valid_coords]
        
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        
        lon_range = (max_lon - min_lon) if max_lon != min_lon else 1.0
        lat_range = (max_lat - min_lat) if max_lat != min_lat else 1.0

        # 가용 가로/세로 길이 (전체 길이 - 양쪽 여백)
        available_width = SCREEN_WIDTH - (2 * MARGIN)
        available_height = SCREEN_HEIGHT - (2 * MARGIN)

        for v in valid_coords:
            # 여백(MARGIN)을 시작점으로 하고, 가용 영역 내에서 정규화된 위치 계산
            x = MARGIN + ((v['lon'] - min_lon) / lon_range) * available_width
            # 위도는 반전 (위도가 높을수록 화면 위쪽)
            y = MARGIN + (1.0 - (v['lat'] - min_lat) / lat_range) * available_height
            
            config_data["stations"][v['name']] = {"x": int(x), "y": int(y)}

    # ---------------------------------------------------------
    # 2. 노선망(Topology) 데이터 처리 (연결 순서 보존)
    # ---------------------------------------------------------
    print("\n--- [Step 2] 노선 데이터 연결 상태 확인 ---")
    try:
        df_routes = pd.read_csv("국토교통부_도시철도 전체노선_20251211.csv", encoding='utf-8')
    except:
        df_routes = pd.read_csv("국토교통부_도시철도 전체노선_20251211.csv", encoding='cp949')

    found_in_routes = set()
    target_lines = ["1호선", "3호선", "4호선", "5호선", "6호선"]
    for line in target_lines:
        line_data = df_routes[df_routes['노선명'].str.contains(line, na=False)].sort_values('순번')
        
        filtered_route = []
        for _, row in line_data.iterrows():
            for target in JONGNO_STATIONS:
                if target in row['역명']:
                    filtered_route.append(target)
                    found_in_routes.add(target)
                    break
        
        if len(filtered_route) > 1:
            # 역 중복(특히 동일 역이 여러 순번 행에 등장)을 제거하되,
            # 노선 순서(연결 순서) 자체는 보존한다.
            seen = set()
            deduped_route = []
            for st in filtered_route:
                if st in seen:
                    continue
                seen.add(st)
                deduped_route.append(st)

            if len(deduped_route) > 1:
                config_data["lines"][line] = deduped_route

    report_missing(list(found_in_routes), JONGNO_STATIONS, "노선 데이터(csv)")

    # ---------------------------------------------------------
    # 3. 시간대별 수요(OD) 데이터 처리 (데이터 통합)
    # ---------------------------------------------------------
    print("\n--- [Step 3] 수요 데이터 분석 상태 확인 ---")
    try:
        df_demand = pd.read_csv("서울시 지하철 호선별 역별 시간대별 승하차 인원 정보.csv", encoding='utf-8')
    except:
        df_demand = pd.read_csv("서울시 지하철 호선별 역별 시간대별 승하차 인원 정보.csv", encoding='cp949')

    found_in_demand = set()
    demand_list = []
    for target in JONGNO_STATIONS:
        match_df = df_demand[df_demand['지하철역'].str.contains(target, na=False)]
        if not match_df.empty:
            found_in_demand.add(target)
            sum_row = match_df.sum(numeric_only=True)
            sum_row['지하철역'] = target
            demand_list.append(sum_row)
    
    report_missing(list(found_in_demand), JONGNO_STATIONS, "수요 데이터(csv)")

    if demand_list:
        df_grouped = pd.DataFrame(demand_list)
        for hour in range(5, 24):
            hr_str = f"{hour:02d}"
            board_col = f"{hr_str}시-{hour+1:02d}시 승차인원"
            alight_col = f"{hr_str}시-{hour+1:02d}시 하차인원"
            
            hr_data = {"spawn_rates": {}, "dest_probs": {}}
            if board_col in df_grouped.columns:
                total_alight = df_grouped[alight_col].sum() if alight_col in df_grouped.columns else 0
                for _, row in df_grouped.iterrows():
                    st = row['지하철역']
                    hr_data["spawn_rates"][st] = float(row[board_col]) / 60.0
                    hr_data["dest_probs"][st] = float(row[alight_col]) / total_alight if total_alight > 0 else 0.0
            config_data["demand"][hr_str] = hr_data

    # 결과 저장
    with open("jongno_config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=4)
    print("\n✨ 전처리 완료! 'jongno_config.json'이 업데이트되었습니다.")

if __name__ == "__main__":
    preprocess_all_data()