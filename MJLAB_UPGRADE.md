# mjlab 업그레이드 기록: v1.1.1에서 v1.6.0으로

과제 보너스(Part 4)를 위해 이 레포를 mjlab v1.6.0에서 돌게 만드는 기록이에요.

- 기준 코드: 교수님 커밋 `6a7fe88`
- 브랜치: `latest-mjlab-v1.6.0`
- 대상 버전: v1.6.0. 과제 공개일이 2026-08-27이고, 그 전 마지막 mjlab 릴리스가 v1.6.0(2026-08-09)이에요.
- 원칙: 학습 동작은 그대로 두고 API만 새 버전에 맞춰요. 수정은 최소로 해요. 동작이 달라질 수 있는 곳은 표의 "동작 변화" 칸에 적어요.

## 진행 상태

| 단계 | 상태 |
|---|---|
| 1. 정적 대조 | 완료 |
| 2. 코드 수정 | 완료. `uv.lock`도 다시 만듦 |
| 3. 바리스타 실행 확인 | 진행 중. 설치와 baseline 학습 완료. 17번을 고친 뒤 baseline 재학습 필요 |

## 1. 의존성 버전

"바꿀 값"은 mjlab v1.6.0의 `uv.lock`에 적힌 버전이에요. `pyproject.toml`에 반영했어요.

| 패키지 | 예전 레포 | v1.6.0 요구 | 바꾼 값 |
|---|---|---|---|
| mjlab | git tag `v1.1.1` | - | git tag `v1.6.0` |
| mujoco | `==3.5.0` | `~=3.11.0` | `==3.11.0` |
| mujoco-warp | git rev `fc91589` | `~=3.11.0` | `==3.11.0`, PyPI |
| warp-lang | `==1.12.0.dev20260206`, nvidia index | `>=1.14.0` | `==1.14.0`, nvidia index |
| rsl-rl-lib | mjlab이 `4.0.1`로 고정 | `==5.4.2` | mjlab이 고정 |
| torch | `==2.9.0`, cu126 | `>=2.7.0` | 그대로 |
| 새 패키지 | - | mjviser, imageio-ffmpeg, numpy | mjlab이 설치 |

## 2. 정적 검사로 찾은 문제

`scripts/check_mjlab_api.py`의 결과예요. mjlab이나 rsl_rl을 쓰는 43개 파일에서 참조 549개, 호출 296개, 하위 클래스 8개를 검사했어요. 경로의 `src/in_hand_rotation_mjlab/`은 생략했어요.

| # | 파일 | 줄 | 깨지는 이유 | 고친 방법 | 동작 변화 | 상태 |
|---|---|---|---|---|---|---|
| 1 | `robots/leap_hand/` 의 `leap_left_constants.py`, `leap_left_custom_constants.py`, `leap_right_constants.py` | 16, 16, 21 | `mjlab.utils.os.update_assets` 삭제 | 같은 동작의 함수를 `leap_right_constants.py`로 옮김. 나머지 두 파일은 거기서 import | 없음 | 고침 |
| 2 | `robots/leap_hand/leap_right_constants.py` | 19 | `DelayedActuatorCfg` 삭제. 지연 설정이 `ActuatorCfg`의 필드가 됨 | 래퍼를 없애고 같은 지연 값을 `IdealPdActuatorCfg`에 직접 넣음. 지연 상수는 그대로 | 없음. v1.4부터 지연이 속도와 토크 목표에도 걸리지만, 이 PD 제어에서는 둘 다 0이라 영향이 없어요 | 고침 |
| 3 | `tasks/hand_cube/hand_cube_env_cfg.py` | 625 | `sync_actuator_delays` 삭제 | mjlab v1.1.1의 함수를 `mdp/events.py`로 옮김. 지연 액추에이터를 고르는 조건만 `has_delay`로 바꿈. env cfg는 옮긴 함수를 부름 | 없음. 16개 모터는 한 그룹으로 묶여서 `set_lags` 한 번이 그룹 전체에 적용돼요 | 고침 |
| 4 | `tasks/hand_cube/hand_cube_env_cfg.py` | 535~602 | `randomize_field` 삭제, `EventTermCfg`의 `domain_randomization` 인자 삭제 | 필드별 새 함수로 바꿈. `body_ipos`는 `dr.body_com_offset`, `dof_frictionloss`는 `dr.joint_friction`, `dof_damping`은 `dr.joint_damping`, `dof_armature`는 `dr.joint_armature`. 범위, 분포, 연산은 그대로. `body_ipos`는 나중에 17번 때문에 `body_com_offset_no_recompute`로 다시 바꿈 | armature는 새 함수가 바꾼 뒤 관련 물리량을 다시 계산해요. 예전 함수는 다시 계산하지 않았어요. 시작할 때 한 번만 일어나요. COM은 17번 참고. friction과 damping은 같아요 | 고침 |
| 5 | `tasks/hand_cube/hand_cube_env_cfg.py` | 614 | `randomize_pd_gains` 삭제 | `dr.pd_gains`, 인자 같음 | 없음 | 고침 |
| 6 | `tasks/hand_cube/hand_cube_env_cfg.py` | 14 | `TerrainImporterCfg` 삭제 | `TerrainEntityCfg`, 인자 같음 | 없음 | 고침 |
| 7 | `tasks/hand_cube/mdp/events.py` | 449 | `DelayedActuator` 삭제 | `set_actuator_effort_limits`에서 지연 래퍼를 벗기는 코드만 지움. 새 버전은 그룹 파라미터를 같은 메모리로 공유해서 `set_effort_limit`이 그대로 반영돼요 | 없음 | 고침 |
| 8 | `tasks/hand_cube/config/` 의 `leap_left`, `leap_left_custom`, `leap_right` 의 `rl_cfg.py` | 10, 17 | `RslRlModelCfg`의 `stochastic`, `init_noise_std` 삭제 | actor는 `distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.7, "std_type": "scalar"}`. critic은 두 인자만 지움. 기본값이 `None`이라 결정적 출력이에요 | 없음 | 고침 |
| 9 | `tasks/hand_cube/mdp/commands.py` | 56, 96, 151 | `CommandTerm._update_command`에 `env_ids` 인자가 생김. 이제 리셋 때도 불림 | 인자를 추가함. `InHandRotationDirectionCommand`는 매 스텝 회전량을 더하므로, 리셋 호출에서는 더하지 않게 함 | 없음 | 고침 |
| 10 | `scripts/train.py` | 151 | `VideoRecorder`의 `log_to_wandb`, `wandb_log_key` 삭제 | 두 인자를 지움 | `--video True`일 때 학습 영상이 W&B에 자동으로 올라가지 않고 로컬에만 저장돼요 | 고침 |
| 11 | `sim2sim/native/actions.py` | 8 | `DelayedActuatorCfg` 삭제 | 래퍼를 벗기는 코드를 지움 | 없음 | 고침 |

## 3. 정적 검사로 못 찾는 문제

소스를 직접 읽고 찾았어요.

| # | 파일 | 문제 | 고친 방법 | 동작 변화 | 상태 |
|---|---|---|---|---|---|
| 12 | `robots/leap_hand/leap_right_constants.py` | `CollisionCfg`의 `contype`, `conaffinity`, `condim`, `priority`가 필수가 됨 | 수정 필요 없음. 원래 설정에 네 값이 모두 있어요. `priority`는 설정 끝부분에 `".*"` 기본값과 함께 있어요. 세 XML의 충돌 geom 71개도 모든 패턴에 걸려요. 처음에 `priority`가 없다고 잘못 보고 한 줄을 넣었다가 지웠어요. 5번 참고 | 없음 | 문제 아님 |
| 13 | `scripts/train.py`, `scripts/play.py`, `scripts/record_replay_trajectory.py`, `policy_server/loader.py`, `sim2sim/policy.py` | rsl_rl 5에 넘기는 설정 형식이 바뀜. mjlab v1.6.0은 None 값 설정을 지워주는 `MjlabOnPolicyRunner`를 씀 | rsl_rl의 `OnPolicyRunner` 대신 `mjlab.rl.MjlabOnPolicyRunner`를 씀. 생성자와 메서드는 같아요 | 없음 | 고침 |
| 14 | 체크포인트 | rsl_rl 5에서 노이즈 파라미터 이름이 `std`에서 `distribution.std_param`으로 바뀜. 예전 체크포인트는 그대로 못 읽어요. 교수님 체크포인트와 우리 baseline 둘 다 해당돼요 | 결정 필요: 읽을 때 이름만 바꾸는 작은 변환을 넣을지 | 없음. 재생할 때는 평균 행동만 써요 | 결정 대기 |
| 15 | 물리 엔진 | MuJoCo가 3.5에서 3.11로 바뀜 | 코드 수정은 없어요. grasp cache를 v1.6.0에서 다시 만들어 비교했어요. 통과 수는 7418에서 7762로 조금 늘었고, 큐브 위치 차이는 0.3 mm 이하, 기울기 차이는 0.03°, 관절 차이는 최대 0.002 rad였어요. 그래서 예전 cache를 그대로 써요 | 4번 표의 baseline 비교 참고 | 확인함 |
| 16 | 큐브 크기 랜덤화 뒤의 충돌 경계값 | 레포의 큐브 크기 함수는 크기만 바꾸고 `geom_rbound`, `geom_aabb`는 그대로 둬요. 새 mjlab의 `dr.geom_size`는 이 값도 다시 계산해요 | 수정 없음. 두 엔진 모두 같은 기본 걸러내기 설정에서 이 값을 같은 방식으로 써요. 그래서 예전 동작과 같아요 | 없음 | 확인함 |
| 17 | `tasks/hand_cube/hand_cube_env_cfg.py`, `tasks/hand_cube/mdp/events.py` | 리셋 때 저장하는 기준 자세가 바뀜. 회전 보상의 자세 게이트, `cube_pose_deviation` 종료, 위치·기울기·`rotation_progress` 지표는 `reset()`에서 큐브 자세를 읽어 기준으로 저장해요. v1.6.0의 `dr.body_com_offset`은 리셋 이벤트가 끝난 뒤 `set_const`를 부르고, `set_const`는 마지막에 몸체 자세를 다시 계산해요. 그래서 새 시작 자세가 저장돼요. v1.1.1은 다시 계산하지 않아서 직전 에피소드 끝 자세가 저장됐어요. 과제 설정값은 v1.1.1 동작에서 맞춰졌어요 | `mdp/events.py`에 `body_com_offset_no_recompute`를 추가함. `dr.body_com_offset`을 그대로 부르지만 `set_const`는 부르지 않아요. `dr_cube_com`이 이 함수를 써요 | 없음. v1.1.1 동작으로 되돌림. 측정은 4번 표의 "리셋 기준 자세 측정" | 고침 |

## 4. 확인 결과

바리스타에서는 GPU 1을 써요. 2026-09-29에 사용자가 요청했어요. GPU 2와 3은 Cosmos 작업이 쓰고 있어요.

| 항목 | 방법 | 결과 | 링크 |
|---|---|---|---|
| 정적 검사 다시 실행 | `scripts/check_mjlab_api.py` | 2번 표의 문제가 모두 사라짐 | - |
| 컴파일 검사 | 수정한 파이썬 파일에 `compile()` | 처음에는 `ast.parse`로 문장 구조만 봐서 인자 중복을 못 잡았어요. `compile()`로 바꾼 뒤 16개 모두 통과 | - |
| `uv.lock` 재생성 | `uv lock`, 설치 없음 | 완료. 141개 패키지 중 7개 변경: mjlab 1.6.0, mujoco 3.11.0, mujoco-warp 3.11.0, warp-lang 1.14.0, rsl-rl-lib 5.4.2 갱신, mjviser 0.0.14, imageio-ffmpeg 0.6.0 추가. torch는 그대로 | - |
| 설치 | 바리스타 `setup_env.sh`. dry run 확인 후 `uv sync --locked` | 완료. 128개 설치, `.venv` 7.0 GB. torch 2.9.0+cu126에서 CUDA 사용 가능, mujoco 3.11.0, mjlab 1.6.0. 로그는 바리스타 `/data/hyeonahj/tools/robot-hand-env/logs/setup_20260929_161618.log` | - |
| 작업 목록 출력 | `scripts/list_envs.py` | 처음에는 12번 문제로 실패, 고친 뒤 성공. LEAP task 3개 등록. 로그는 바리스타 `robot-hand-env/logs/list_envs_20260929_190702.log` | - |
| zero 에이전트 재생 | `scripts/play.py --agent zero`, GPU 1, 4 env, 2분 | 환경 생성, 관리자 설정, grasp cache 7418개 읽기, 뷰어 시작까지 에러 없음. 로그는 바리스타 `robot-hand-env/logs/play_zero_20260929_191409.log` | - |
| random 에이전트 재생 | `scripts/play.py --agent random`, GPU 1, 4 env, 2분 | 첫 실행은 GPU 커널 컴파일로 2분이 지나 확인 못함. 다시 실행해서 환경 생성과 뷰어 시작까지 에러 없음. 도는 동안 GPU 1 사용률 20~24%, 메모리 590 MB로 꾸준해서 스텝이 실제로 돈 것을 확인. 로그는 바리스타 `robot-hand-env/logs/play_random_20260929_192744.log` | - |
| 물리 옵션 경고 | 재생 로그 첫 줄 | 손 XML의 `<option>` 값 `impratio=100`, `implicitfast`, `elliptic`은 손을 장면에 붙일 때 무시돼요. 실제 값은 env 설정이 정해요. `impratio=10`, `cone="elliptic"`을 직접 넣고, 적분기는 두 버전 모두 기본값 `implicitfast`예요. 그래서 예전과 같아요 | - |
| 짧은 학습 | 256 env, 20 iteration, seed 42, GPU 1, 바리스타 RTX 6000 Ada | 정상 종료. 보상 값 정상, NaN 종료 0번. 에피소드 길이 13에서 306 스텝으로 늘어서 v1.1.1 확인 학습과 같은 경향. iteration당 1.5~1.7초. 경고는 tyro 타입 표기 경고와 W&B 안내뿐이고 영향 없음. 로그는 바리스타 `robot-hand-env/logs/train_test256_20260929_193845.log` | [W&B](https://wandb.ai/hyeonah-jung-usc/robot-hand-latest-test/runs/oupu57ii) |
| baseline 학습 | 4096 env, 5000 iteration, seed 42, GPU 1, W&B 프로젝트 `robot-hand-latest`, run 이름 `baseline` | 정상 종료. iteration당 약 2.5초. 마지막 100 iteration 평균을 v1.1.1 baseline과 비교하면, 회전 보상 0.243 대 0.252로 비슷하고 에피소드 길이 391 대 385로 실패는 조금 적어요. 위치 오차 1.17 cm 대 0.95 cm, 기울기 오차 0.175 대 0.145 rad로 커서 `rotation_progress`가 0.176 대 0.317로 낮아요. 커리큘럼은 이 지표로 벌점 가중치를 올려요. 그래서 마지막 가중치가 v1.1.1의 약 65%에서 멈췄어요. 예를 들어 torque는 -0.649 대 -1.0이에요. 원인은 17번이에요. 이 run은 17번을 고치기 전 코드예요. 로그는 바리스타 `robot-hand-env/logs/train_baseline_20260929_210234.log` | [W&B](https://wandb.ai/hyeonah-jung-usc/robot-hand-latest/runs/4fjqzu9p) |
| grasp cache 재생성 | `scripts/collect_hand_cube_grasp_cache.py`, 기본값, GPU 1, 레포 밖에 저장 | 7762개 통과. 예전 cache와 거의 같음. 15번 참고. 파일은 바리스타 `robot-hand-env/grasp_cache_mjlab160/`, 로그는 `robot-hand-env/logs/collect_grasp_cache_20260930_043608.log` | - |
| 고정 조건 평가 | `scripts/eval_hand_cube.py`, v1.6.0 환경, 1024 env, seed 0, 첫 에피소드 400 스텝. v1.1.1 baseline 체크포인트는 14번의 이름 변환으로 읽음 | 기본 마찰에서 v1.6.0 baseline은 yaw 0.329 rad/s, 생존 95.8%예요. v1.1.1 baseline은 0.350 rad/s, 98.6%예요. 결과는 바리스타 `robot-hand-env/eval/` | - |
| 마찰 고정 평가 | 같은 방법, 마찰 0.3~2.0 | 아래 첫 표. 같은 v1.6.0 환경에서 v1.1.1 정책은 모든 마찰에서 생존 97.9% 이상이에요. v1.6.0 정책은 마찰 0.3에서 56.4%예요. 그래서 환경이 아니라 학습된 정책이 달라요. 결과는 `robot-hand-env/eval_friction/`, `eval_friction_v1.1.1/` | - |
| 리셋 기준 자세 측정 | 레포에 넣지 않은 일회성 진단 스크립트. 학습 설정, 256 env, seed 0, zero 행동. 리셋 직후 종료 조건, 지표, 보상이 저장한 큐브 기준 자세와 실제 시작 자세의 차이 | 아래 둘째 표. 세 항목의 값은 같았어요. v1.6.0에서 `set_const`만 끄면 v1.1.1과 같아요. 17번 참고 | - |

마찰 고정 평가, v1.6.0 환경, 17번 수정 전:

| 마찰 | v1.6.0 baseline yaw rad/s | 생존 | v1.1.1 baseline yaw rad/s | 생존 |
|---|---|---|---|---|
| 0.3 | 0.109 | 56.4% | 0.344 | 98.5% |
| 0.5 | 0.308 | 87.8% | 0.359 | 99.1% |
| 1.0 | 0.325 | 97.8% | 0.344 | 98.2% |
| 1.6 | 0.282 | 94.1% | 0.308 | 97.9% |
| 2.0 | 0.232 | 89.0% | 0.278 | 98.1% |

리셋 기준 자세 측정, 두 번째 `env.reset()` 직후 256 env 평균:

| 환경 | 위치 차이 | 기울기 차이 |
|---|---|---|
| v1.1.1 | 1.30 cm | 0.219 rad |
| v1.6.0, 수정 전 | 0.00 cm | 0.000 rad |
| v1.6.0, `set_const`만 끔 | 1.24 cm | 0.198 rad |

v1.1.1과 `set_const`를 끈 v1.6.0은 첫 `env.reset()` 직후 차이가 평균 3.7 m예요. 그래서 모든 env가 첫 스텝에 `cube_pose_deviation`으로 종료되고 다시 리셋돼요. 수정 전 v1.6.0에서는 이 종료가 없어요.

## 5. 에러와 해결

| 날짜 | 단계 | 에러 | 원인 | 해결 |
|---|---|---|---|---|
| 2026-09-29 | 바리스타 작업 목록 출력 | `leap_right_constants.py` 431줄 `SyntaxError: keyword argument repeated: priority` | 원래 충돌 설정 끝부분에 `priority`가 이미 있었어요. 설정 앞부분만 읽고 없다고 판단해서 `priority=0`을 한 줄 더 넣었어요. 검사에 쓴 `ast.parse`는 인자 중복을 못 잡아요 | 넣은 줄을 지웠어요. 충돌 설정은 교수님 코드와 같아졌어요. 검사를 `compile()`로 바꿨어요 |
| 2026-09-30 | baseline 학습 뒤 평가 | 에러는 없지만 v1.6.0 baseline이 낮은 마찰에서 약해요. 마찰 0.3에서 생존 56.4% | 17번. 리셋 기준 자세가 바뀌어서 과제가 더 엄격해졌고, 커리큘럼 벌점도 덜 올라갔어요. grasp cache와 충돌 경계값을 먼저 의심했지만 원인이 아니었어요. 4번 표에서 COM 재계산을 "더 정확해짐"으로만 적고, 기준 자세에 주는 영향을 놓쳤어요 | `dr_cube_com`이 `body_com_offset_no_recompute`를 쓰게 바꿈. baseline 재학습 필요 |

## 6. 정할 것

1. 14번: 예전 체크포인트를 읽는 변환을 넣을지.
2. 보너스 범위: README에 "update the code ... (and then answer the above questions)"라고 되어 있어요. 새 버전으로 baseline과 수정 2개를 다시 돌려야 하는지 교수님이나 조교에게 확인이 필요해요.

## 7. 다시 검사하는 법

설치 없이 파이썬만 있으면 돼요. 문제가 없으면 2번부터 6번까지 모두 "None"이 나와요.

```
curl -L https://github.com/mujocolab/mjlab/archive/refs/tags/v1.1.1.tar.gz | tar -xz
curl -L https://github.com/mujocolab/mjlab/archive/refs/tags/v1.6.0.tar.gz | tar -xz
curl -L https://github.com/leggedrobotics/rsl_rl/archive/refs/tags/v4.0.1.tar.gz | tar -xz
curl -L https://github.com/leggedrobotics/rsl_rl/archive/refs/tags/v5.4.2.tar.gz | tar -xz
python scripts/check_mjlab_api.py \
  --pkg mjlab mjlab-1.1.1 mjlab-1.6.0 \
  --pkg rsl_rl rsl_rl-4.0.1 rsl_rl-5.4.2
```
