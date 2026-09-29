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
| 3. 바리스타 실행 확인 | 대기 |

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
| 4 | `tasks/hand_cube/hand_cube_env_cfg.py` | 535~602 | `randomize_field` 삭제, `EventTermCfg`의 `domain_randomization` 인자 삭제 | 필드별 새 함수로 바꿈. `body_ipos`는 `dr.body_com_offset`, `dof_frictionloss`는 `dr.joint_friction`, `dof_damping`은 `dr.joint_damping`, `dof_armature`는 `dr.joint_armature`. 범위, 분포, 연산은 그대로 | COM과 armature는 새 함수가 바꾼 뒤 관련 물리량을 다시 계산해요. 예전 함수는 다시 계산하지 않았어요. 더 정확해지지만 결과가 조금 다를 수 있어요. friction과 damping은 같아요 | 고침 |
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
| 12 | `robots/leap_hand/leap_right_constants.py` | `CollisionCfg`의 `contype`, `conaffinity`, `condim`, `priority`가 필수가 됨. 지금 `priority`가 없음 | 예전 기본값 `priority=0`만 추가. 세 XML의 충돌 geom 71개가 이미 모든 패턴에 걸려서 다른 수정은 필요 없어요 | 없음 | 고침 |
| 13 | `scripts/train.py`, `scripts/play.py`, `scripts/record_replay_trajectory.py`, `policy_server/loader.py`, `sim2sim/policy.py` | rsl_rl 5에 넘기는 설정 형식이 바뀜. mjlab v1.6.0은 None 값 설정을 지워주는 `MjlabOnPolicyRunner`를 씀 | rsl_rl의 `OnPolicyRunner` 대신 `mjlab.rl.MjlabOnPolicyRunner`를 씀. 생성자와 메서드는 같아요 | 없음 | 고침 |
| 14 | 체크포인트 | rsl_rl 5에서 노이즈 파라미터 이름이 `std`에서 `distribution.std_param`으로 바뀜. 예전 체크포인트는 그대로 못 읽어요. 교수님 체크포인트와 우리 baseline 둘 다 해당돼요 | 결정 필요: 읽을 때 이름만 바꾸는 작은 변환을 넣을지 | 없음. 재생할 때는 평균 행동만 써요 | 결정 대기 |
| 15 | 물리 엔진 | MuJoCo가 3.5에서 3.11로 바뀜 | 코드 수정은 없어요. 체크포인트 재생과 짧은 학습으로 확인해요. 필요하면 grasp cache를 다시 만들어요 | 확인 필요 | 실행 때 확인 |

## 4. 확인 결과

| 항목 | 방법 | 결과 | 링크 |
|---|---|---|---|
| 정적 검사 다시 실행 | `scripts/check_mjlab_api.py` | 2번 표의 문제가 모두 사라짐 | - |
| 문법 검사 | 수정한 파이썬 파일 15개 | 통과 | - |
| `uv.lock` 재생성 | `uv lock`, 설치 없음 | 완료. 141개 패키지 중 7개 변경: mjlab 1.6.0, mujoco 3.11.0, mujoco-warp 3.11.0, warp-lang 1.14.0, rsl-rl-lib 5.4.2 갱신, mjviser 0.0.14, imageio-ffmpeg 0.6.0 추가. torch는 그대로 | - |
| 설치 | `uv sync --dry-run` 후 `uv sync` | 대기 | - |
| 작업 목록 출력 | `scripts/list_envs.py` | 대기 | - |
| random 에이전트 재생 | `scripts/play.py --agent random` | 대기 | - |
| 짧은 학습 | 256 env, 20 iteration | 대기 | - |
| baseline 학습 | 4096 env, 5000 iteration | 보너스 범위 정한 뒤 | - |

## 5. 에러와 해결

아직 없어요.

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
