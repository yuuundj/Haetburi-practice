# archive

작업 중에 임시 폴더에만 있던 기록과, 버전 구조로 바꾸기 직전의 상태를 보존한 곳입니다.
모델 자체는 `robot/<버전>/`에서 다시 만들어지므로, 여기 있는 것은 당시의 기록입니다.

| 경로 | 내용 |
|---|---|
| `2026-09-23_session/analysis/` | 버전별 당시 분석 결과(`sim/analyse.py` 출력)와 데모 부하 기록 |
| `2026-09-23_session/images/` | 원본 STL 투영도, v1 첫 렌더, v1 700 mm 미리보기, v6 하체 근접, v1 데모 동영상 |
| `2026-09-23_session/ankle_search/` | 발목 모터 2개를 발에 넣는 배치 전수 탐색 스크립트와 상위 10개 결과(pickle) |
| `robot_before_variants/` | 버전 폴더로 나누기 전 `robot/`에 있던 v6 출력 (지금의 `robot/v6_split_round/`와 같음) |
| `code_before_variants/` | 버전 구조로 바꾸기 직전의 `tools/`, `sim/`, README |

`v6_draft_rocker_sole_analyse.txt`는 v6을 만드는 도중, 발바닥을 번의 가장 넓은
둘레보다 아래에서 잘랐던 시안의 결과입니다. 발이 흔들의자처럼 구르는 문제가 있어
발바닥 위치를 바꿨고, 그 시안은 버전으로 남기지 않았습니다.

`ankle_search/ranked_top10.pkl`은 발 맞춤 함수의 버그를 고치기 전에 계산한 순위라,
발 크기 값은 지금 코드로 계산한 값과 다릅니다.
