# 실험 기록: baseline, SoftGate, WideFriction

과제 Part 3의 수정 2개를 기록해요. 수정마다 관찰과 이유, 가설, 바꾼 것, 확인한 것, 반증 기준을 적어요. 결과 칸은 학습과 평가가 끝나면 채워요.

## 공통 조건

| 항목 | 값 |
|---|---|
| 기반 task | `Mjlab-Leap-Left-HandCube-Rotate` |
| 학습 예산 | 4096 env × 32 스텝 × 5000 iteration = 655,360,000 env 스텝 |
| seed | 42 |
| PPO 설정 | 세 run 모두 baseline의 `rl_cfg.py`와 같음 |
| 제어 주기 | 0.005 s × decimation 10 = 0.05 s, 20 Hz |
| 에피소드 길이 | 20 s = 400 스텝 |
| 장비와 버전 | blackcoffee RTX 4090 GPU 1 하나, mjlab v1.1.1 |
| W&B 프로젝트 | `hyeonah-jung-usc/robot-hand` |

SoftGate와 WideFriction은 같은 GPU에서 동시에 학습했어요. 그래서 iteration당 시간은 baseline보다 길어요. 학습 스텝 수는 세 run 모두 같아요.

## 학습에 실제 쓰인 설정

각 run의 `params/env.yaml`, `params/agent.yaml`에서 확인한 값이에요.

| run | 회전 보상 게이트 | 손-큐브 마찰 범위 | seed | iteration | rollout | W&B run | 커밋 |
|---|---|---|---|---|---|---|---|
| baseline | `step` | 0.6~1.4 | 42 | 5000 | 32 | `dpjn57x0` | `6a7fe88` |
| SoftGate | `soft` | 0.6~1.4 | 42 | 5000 | 32 | `czqifh6r` | `e49a754` |
| WideFriction | `step` | 0.4~1.8 | 42 | 5000 | 32 | `dgn96zo9` | `e6cc6e8` |

## Baseline에서 본 현상

학습 로그의 iteration 구간 평균이에요. 도메인 랜덤화와 관측 노이즈가 켜진 학습 중 값이라 참고용이에요.

| 지표 | 2400~2499 | 4900~4999 |
|---|---|---|
| 평균 보상 | 3.304 | 3.564 |
| `rotation_progress` | 0.346 | 0.317 |
| 행동 노이즈 std | 3.15 | 7.67 |
| 기울기 오차, rad | 0.131 | 0.145 |
| 위치 오차, m | 0.0092 | 0.0095 |

- 보상은 올랐는데 과제 지표인 `rotation_progress`는 떨어졌어요.
- 행동 노이즈 std가 두 배 넘게 커졌어요. 행동은 [-1, 1]로 잘려요. 그래서 이 값이면 학습 중 샘플 대부분이 양 끝으로 잘릴 것으로 추정해요. 실제 비율은 평가에서 측정해요.
- 원인은 아직 확정하지 않았어요.

## 수정 1: SoftGate, 트랙 A 보상

### 관찰과 이유

- 회전 보상에는 자세 게이트가 곱해져요. baseline의 `step` 게이트는 위치 오차 2 cm, 기울기 0.35 rad 안이면 1.0이고 밖이면 0.1이에요.
- 과제 지표 `rotation_progress`는 허용 범위 안에서도 오차가 커지면 낮아져요.
- 그래서 보상은 같은데 실제 안정성은 다른 행동이 있을 수 있어요. 위 표에서 보상과 지표가 반대로 움직인 것과 맞는 설명이에요.

### 가설

허용 범위 안의 작은 자세 오차까지 보상에 반영하면, 회전 속도를 유지하면서 위치 오차, 기울기 오차, 자세 이탈이 줄어든다.

### 바꾼 것

경로의 `src/in_hand_rotation_mjlab/tasks/hand_cube/`는 생략했어요.

| 파일 | 내용 |
|---|---|
| `mdp/rewards.py` | 회전 보상 게이트에 `drift_mode="soft"`를 추가. 기존 `step`, `exp`는 그대로 |
| `config/leap_left_mods/env_cfgs.py` | baseline 설정에서 `drift_mode`만 `"soft"`로 바꿈 |
| `config/leap_left_mods/rl_cfg.py` | baseline PPO 설정에서 `experiment_name`만 바꿈 |
| `config/leap_left_mods/__init__.py` | task `Mjlab-Leap-Left-HandCube-Rotate-SoftGate` 등록 |

게이트 식이에요. 0.1과 1.0은 기존 파라미터 `drift_outside_factor`와 `drift_inside_factor`예요. 두 점수는 `rotation_progress`의 안정성 점수와 같은 식이에요.

```
pos_score  = clamp(1 - pos_error / 0.02, 0, 1)
tilt_score = clamp(1 - tilt_error / 0.35, 0, 1)
factor     = 0.1 + 0.9 * pos_score * tilt_score
```

| 위치 오차 | 기울기 오차 | baseline 배율 | SoftGate 배율 |
|---|---|---|---|
| 0 cm | 0 rad | 1.0 | 1.0 |
| 1 cm | 0.15 rad | 1.0 | 0.357 |
| 2 cm 이상 | 0 rad | 경계에서 1.0, 넘으면 0.1 | 0.1 |

### 확인한 것

- baseline과 설정을 비교하니 다른 값은 `drift_mode`와 `experiment_name` 두 개뿐이었어요. 설정을 만들 때마다 새로 생기는 함수 객체 `spec_fn`의 주소 차이는 제외했어요.
- 관측 차원이 actor 320, critic 83으로 baseline과 같아요.
- 256 env, 20 iteration 짧은 학습에서 NaN과 에러가 없었어요.

### 반증 기준

위치와 기울기 오차는 줄었지만 큐브를 거의 돌리지 않으면, "안정적인 회전을 개선했다"는 가설을 지지하지 않아요.

### 결과

학습과 평가가 끝나면 채워요.

## 수정 2: WideFriction, 트랙 C 도메인 랜덤화

### 관찰과 이유

- 손끝은 마찰력으로 큐브를 돌려요.
- 마찰이 낮으면 큐브가 미끄러져 밀리거나 떨어져요. 마찰이 높으면 손끝이 붙어서 회전이 막혀요.
- 그래서 정책은 마찰에 맞게 접촉 방식을 바꿔야 해요. baseline은 마찰 0.6~1.4에서만 배웠어요.

### 가설

더 넓은 마찰 범위 0.4~1.8에서 학습하면, 미끄러운 큐브와 잘 붙는 큐브에서도 안정적으로 회전시키는 정책이 된다.

### 바꾼 것

| 파일 | 내용 |
|---|---|
| `config/leap_left_mods/env_cfgs.py` | `events["dr_shared_contact_friction"].params["friction_range"]`를 (0.6, 1.4)에서 (0.4, 1.8)로 바꿈 |
| `config/leap_left_mods/rl_cfg.py` | baseline PPO 설정에서 `experiment_name`만 바꿈 |
| `config/leap_left_mods/__init__.py` | task `Mjlab-Leap-Left-HandCube-Rotate-WideFriction` 등록 |

### 확인한 것

- baseline과 설정을 비교하니 다른 값은 `friction_range`와 `experiment_name` 두 개뿐이었어요. 같은 이유로 `spec_fn`의 주소 차이는 제외했어요.
- 관측 차원이 actor 320, critic 83으로 baseline과 같아요.
- 256 env, 20 iteration 짧은 학습에서 NaN과 에러가 없었어요.

### 평가 계획

마찰을 고정한 조건에서 baseline과 비교해요. 낮음, 보통, 높음 세 조건이에요. 낮음과 높음은 두 정책 모두의 학습 범위 밖 값으로 정해요. 정확한 값은 평가 스크립트를 만들 때 확정해요.

### 반증 기준

- 낮은 마찰과 높은 마찰에서 baseline보다 나아지지 않으면 가설을 지지하지 않아요.
- 모든 마찰에서 보통 조건 성능까지 떨어지면, "범위가 너무 넓어서 학습이 어려워졌다"로 해석해요.

### 결과

학습과 평가가 끝나면 채워요.

## 실험하지 않은 관찰: 행동 노이즈

baseline 후반에 행동 노이즈 std가 3.15에서 7.67로 커졌어요. entropy 계수는 학습 내내 0.003으로 고정이에요. 후반에 탐색이 지나쳐 정밀한 제어를 방해한다는 가설을 세울 수 있어요. 시간 때문에 실험하지 않고 종합 논의의 다음 단계로 남겨요.

## 다른 기록

- 환경 설치 중 생긴 문제: 레포 밖 `robot-hand/LOG.md`
- mjlab v1.6.0 업그레이드: `latest-mjlab-v1.6.0` 브랜치의 `MJLAB_UPGRADE.md`
