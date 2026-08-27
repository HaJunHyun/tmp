# 이 인스턴스 셋업 노트 (RQL)

## 환경
- Python env: `/venv/main` — `source /venv/main/bin/activate`
- JAX 0.11.1 + `jax-cuda12-plugin` (CUDA 12.9 wheel), GPU 8x RTX 4090 (cc 8.9) 인식 확인
- ogbench 1.2.0, gymnasium 0.29.1, mujoco 3.12.0, flax 0.12.9, distrax 0.1.9, optax 0.2.8
- 고정 버전 전체: `requirements.lock.txt`

## 원본 requirements 대비 바뀐 점
1. `wandb==0.18.7`로 고정. 최신 wandb(0.29)는 `wandb.Settings(start_method=..., _disable_stats=...)`
   를 더 이상 받지 않아 `utils/log_utils.py:setup_wandb`가 pydantic ValidationError로 죽습니다.
2. requirements.txt에 없지만 코드가 import 하는 패키지 추가: `einops`, `opencv-python-headless`,
   `pillow`, `tqdm`, `optax`.
3. `utils/log_utils.py` / `envs/log_utils.py`의 `np.reshape(v, newshape=...)` → `np.reshape(v, ...)`.
   numpy 2.x에서 `newshape` 키워드가 제거됨 (영상 로깅 `--video_episodes > 0` 일 때만 실행되는 경로).

## 데이터셋
공식 호스트 `rail.eecs.berkeley.edu`가 이 인스턴스에서 접속 불가(타임아웃)라
ogbench 내장 다운로더가 실패합니다. HuggingFace 미러에서 받는 헬퍼를 추가했습니다:

```bash
python download_ogbench_data.py --list                    # 미러에 있는 목록
python download_ogbench_data.py humanoidmaze-large-navigate-v0
```
파일은 `~/.ogbench/data/` 에 저장되며, 그 뒤 `ogbench.make_env_and_datasets`가 그대로 씁니다.
미러 repo: `zhouzypaul/ogbench_datasets` (state 기반 데이터셋 전부 + visual-scene).
※ 논문의 100M 데이터셋(cube-quadruple / puzzle-4x4)은 이 미러에 없고 README의 wget 주소도
   같은 막힌 호스트입니다.

`antmaze-medium-navigate-v0`는 이미 받아두었습니다.

## 실행
```bash
source /venv/main/bin/activate
export MUJOCO_GL=egl
wandb login          # 또는 wandb offline / export WANDB_API_KEY=...

python main.py --agent=agents/rql.py \
  --env_name=humanoidmaze-large-navigate-singletask-v0 \
  --agent.alpha=10 --agent.expectile=0.9 --agent.ensemble_ct=10 \
  --agent.rho=0.0 --agent.h=1 --agent.discount=0.995 \
  --offline_steps=1000000 --online_steps=0 --agent.batch_size=256
```
환경별 튜닝된 하이퍼파라미터는 `hyperparameters.sh` 참고.

- `main.py`의 `setup_wandb(...)`가 `mode='online'`을 하드코딩하므로 `WANDB_MODE=offline`
  환경변수는 무시됩니다. 오프라인으로 돌리려면 `wandb offline`을 쓰거나 로그인하세요.
- 여러 GPU 중 하나만 쓰려면 `CUDA_VISIBLE_DEVICES=0`.
- 영상 로깅(`--video_episodes>0`)은 OpenGL/EGL 라이브러리가 필요합니다.
  현재 미설치 상태이니 필요하면 `install-display-drivers` 실행 후 사용하세요.
- `/workspace`는 이 인스턴스에서 볼륨이 아닙니다(recycle/destroy 시 소실). 중요한 결과는 외부로 백업하세요.

## 스모크 테스트 (통과)
`antmaze-medium-navigate-singletask-v0`, 200 offline steps + eval 2 episodes 정상 완료.
학습 속도 정상(컴파일 후 ~25 it/s), GPU matmul/환경 스텝/평가 모두 동작.

## 추가 패치: `--eval_at`
`main.py`에 `--eval_at` 플래그를 추가했습니다. 지정한 스텝에서만 평가하며, 주면 `--eval_interval`을 무시합니다.
```bash
--eval_at=800000,900000,1000000 --eval_episodes=50
```
지표는 `<save_dir>/eval.csv`의 `evaluation/success` 컬럼에서 뽑습니다:
- 800k/900k/1M 3개 행 평균
- 1M 행 단일값

## 스윕 실행 (320 runs = 8 domain × 5 task × 8 seed)

```bash
source /venv/main/bin/activate
cd /workspace/rql
python launch_sweep.py --concurrency=48          # 실행 (중단 후 재실행하면 이어서 진행)
python launch_sweep.py --status                  # 진행 상황
python collect_results.py                        # 결과 집계
python collect_results.py --by task --csv runs.csv
```

- 결과: `/workspace/rql_results/` (`runs/<run_id>/**/eval.csv`, `logs/<run_id>.log`, `status.json`, `sweep.log`)
- 하이퍼파라미터는 `launch_sweep.py`의 `DOMAINS` 표 = hyperparameters.sh 그대로.
  `--sparse`는 scene-play / puzzle-3x3-play, `discount=0.995`는 antmaze-giant / humanoidmaze-*.
- 평가는 `--eval_at=800000,900000,1000000`, `--eval_episodes=50` → run당 3회.
- 자식 프로세스 환경변수(런처가 자동 설정): 컴파일 캐시, `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
  `--xla_gpu_force_compilation_parallelism=1`, `WANDB_MODE=disabled`.
  **컴파일 캐시 워밍업이 없으면 동시 실행 시 `Failed to launch ptxas`로 다수가 죽습니다** —
  런처가 시작 시 도메인별로 자동 워밍업합니다.
- 중단: `pkill -f launch_sweep.py` (SIGTERM → 실행 중인 자식도 정리). 재실행하면 완료된 run은 건너뜁니다.
- 실패한 run은 최대 2회 자동 재시도, 그래도 실패하면 `status.json`의 `failed_ids`에 남습니다.
