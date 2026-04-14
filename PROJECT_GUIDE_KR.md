# Mini Metro RL 프로젝트 가이드 (한국어)

이 문서는 이 프로젝트를 처음 보는 사람도 빠르게 이해하고, 새로운 문제(MDP)를 정의해 실험하고, 필요한 파일만 정리해 GitHub에 업로드할 수 있도록 만든 실전 안내서입니다.

## 1. 프로젝트 한눈에 보기

이 프로젝트는 `Mini Metro` 스타일 시뮬레이션을 Python + pygame으로 구현한 뒤, 강화학습 실험이 가능하도록 환경을 분리한 구조입니다.

- 수동 플레이: `src/main.py`
- 일반 프로그래밍 API 환경: `src/env.py` (`MiniMetroEnv`)
- 종로 수요 기반 RL 환경: `src/jongno_env.py` (`JongnoMetroEnv`)
- 공통 게임 엔진/규칙: `src/mediator.py`
- Task 선택 실행기: `src/task_catalog.py`, `src/task_runner.py`

핵심 아이디어는 **게임 엔진(`Mediator`)은 유지**하고, 환경/보상/상태/행동 정의를 바꿔가며 서로 다른 MDP를 실험하는 것입니다.

## 2. 지금 가능한 실행 방식

- 수동 플레이:
  - `python src/main.py`
- 테스트:
  - `python -m unittest -v`
- Task 목록:
  - `python src/task_runner.py --list`
- 특정 Task 실행:
  - `python src/task_runner.py --task jongno_dispatch --policy random --steps 300`
- 환경 파라미터만 바꿔 실행:
  - `python src/task_runner.py --task jongno_dispatch --env-overrides "{\"max_budget\": 10}"`
- 사용자 정의 정책:
  - `python src/task_runner.py --task jongno_dispatch --policy my_policy_module:act`

## 3. MDP를 바꾸는 방법 (문제/보상/상태/행동)

MDP 변경은 보통 아래 4가지를 수정합니다.

### A) 문제 정의(환경 설정) 변경

`src/task_catalog.py`에서 task builder를 추가/수정합니다.

```python
def _build_my_task(overrides: Dict[str, Any]) -> JongnoMetroEnv:
    config = {
        "max_budget": 15,
        "dt_ms": 1000,
        "hour_advance_per_step": 1.2,
        "demand_spawn_scale": 0.007,
        "reward_waiting_weight": 0.2,
        "score_reward_weight": 1.0,
    }
    config.update(overrides)
    return JongnoMetroEnv(**config)
```

그리고 `TASK_SPECS`에 등록합니다.

```python
"my_task": TaskSpec(
    task_id="my_task",
    description="내가 정의한 문제",
    env_factory=_build_my_task,
    default_steps=800,
),
```

### B) 행동(action) 변경

행동 정의를 바꾸려면 주로 `src/jongno_env.py`의 아래를 수정합니다.

- `action_space`
- `step()` 내부 action 해석 로직
- `info["action_ok"]`, 패널티 계산

행동 종류를 확장하려면 `Mediator.apply_action(...)`와 같이 연동해야 합니다.

### C) 상태(observation) 변경

`src/jongno_env.py`의 `_get_obs()`를 수정해 관측 벡터 구성요소를 바꿉니다.

예시:
- 역별 대기 승객 수
- 노선별 운행 열차 수
- 시간 인덱스
- 추가 가능: 혼잡 지수, 최근 보상 이동평균, 이벤트 플래그

상태 차원을 바꾸면 `observation_space`도 반드시 함께 수정해야 합니다.

### D) 보상(reward) 변경

`src/jongno_env.py`의 `step()` 마지막 reward 계산부를 수정합니다.

현재 예시 형태:
- `score_delta` 보상
- `waiting_ratio_sum` 패널티
- invalid/budget 패널티

추천 방식:
- 항목별 가중치를 명시적으로 분리 (`w_score`, `w_wait`, `w_overflow`, `w_switch`)
- `info`에 reward 구성요소를 함께 반환 (디버깅/튜닝 쉬움)

## 4. 새 MDP 추가할 때 체크리스트

- 환경 생성자 인자와 실제 `JongnoMetroEnv.__init__` 인자가 일치하는가?
- action/observation space와 실제 반환값 shape가 일치하는가?
- reward가 너무 sparse 하거나 폭주하지 않는가?
- `python src/task_runner.py --task ...`가 정상 실행되는가?
- `python -m unittest -v` 통과하는가?

## 5. 정책(policy) 추가 방법

### 방법 1) 내장 정책으로 추가

`src/task_runner.py`에 함수 추가 후 `BUILT_IN_POLICIES`에 등록:

```python
def my_policy(env, obs, info) -> int:
    return 0
```

### 방법 2) 외부 모듈 플러그인

예: `src/my_policy_module.py`

```python
def act(env, obs, info) -> int:
    return int(env.action_space.sample())
```

실행:
- `python src/task_runner.py --task jongno_dispatch --policy my_policy_module:act`

## 6. GitHub 업로드용 파일 정리

아래는 **권장 업로드 대상**입니다.

### 반드시 포함

- 코드: `src/`
- 테스트: `test/`
- 실험 설정: `jongno_config.json`
- 의존성/설정: `requirements.txt`, `pyproject.toml`, `environment.yml`
- 문서: `README.md`, `GAME_RULES.md`, `INSTRUCTION_TASK_RUNNER.md`, `PROJECT_GUIDE_KR.md`
- 구조/기록: `ARCHITECTURE.md`, `PROGRESS.md`
- CI: `.github/workflows/test.yml`
- 품질도구: `.pre-commit-config.yaml`, `.gitignore`

### 선택 포함

- 데이터 전처리/시각화 스크립트:
  - `preprocess_jongno.py`
  - `visualize_smoke_log.py`

### 제외 권장

- 로컬 IDE/개인 설정 파일
- 대용량 산출물(로그, 체크포인트, 영상, 임시 결과)
- 민감정보(키/토큰/개인 데이터)

## 7. GitHub 업로드 절차 (처음부터)

```bash
git init
git add .
git commit -m "Initial commit: Mini Metro RL task runner and docs"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

업로드 전 마지막 확인:
- 테스트 통과: `python -m unittest -v`
- README 실행 명령 최신화
- 새 task/policy 문서 반영 여부

## 8. 협업 규칙 제안

- 새 task 추가 시 `src/task_catalog.py` + 문서 + 테스트를 함께 변경
- 보상 변경 시 `info`에 reward 분해값 포함
- 큰 구조 변경 후 `ARCHITECTURE.md`, `PROGRESS.md` 동시 업데이트

---


