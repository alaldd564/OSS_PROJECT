# OSS 과제 결과보고서

## 1. 과제 개요
본 과제는 GitHub를 활용하여 소스코드를 관리하고, scikit-learn 기반의 머신러닝 실습을 통해 OSS 개발 흐름을 경험하는 것을 목표로 한다.

- 과제 주제: Wine 데이터셋 분류 모델 구현
- 데이터셋: `load_wine`
- 문제 유형: 다중 분류(Classification)
- 사용 도구: Python, scikit-learn, GitHub
- 저장소 관리 방식: 개인 GitHub 저장소 + 작업 브랜치 기반 개발 + 단계별 commit 기록

## 2. 구현 요구사항 충족 여부
아래 요구사항을 모두 반영했다.

- 데이터셋 로드
- train/test split
- 머신러닝 모델 학습
- 예측 및 정확도 출력

실행 파일은 [oss_ml_wine_classification.py](oss_ml_wine_classification.py) 이다.

## 3. 실험 설계
단일 모델만 제출하는 방식보다, 최소 2개 이상의 모델을 비교하고 하이퍼파라미터 탐색까지 포함해야 과제 완성도가 높아진다. 따라서 다음과 같이 구성했다.

- Logistic Regression: 표준화 포함 베이스라인 모델
- Random Forest: 비선형 분류 성능 비교용 모델
- Random Forest + GridSearchCV: 하이퍼파라미터 탐색 기반 개선 모델
- 5-fold Stratified Cross Validation: 과적합 여부를 더 안정적으로 확인

## 4. 데이터셋 정보
- 샘플 수: 178
- 특성 수: 13
- 클래스 수: 3
- 학습/평가 분리: train 142, test 36

## 5. 주요 결과
| 모델 | Test Accuracy | CV Accuracy | 비고 |
| --- | ---: | ---: | --- |
| LogisticRegression | 0.9722 | 0.9791 | 표준화 기반 베이스라인 |
| RandomForest | 1.0000 | 0.9791 | 테스트셋 완전 정분류 |
| RandomForest(GridSearchCV) | 1.0000 | 0.9862 | 탐색 기반 개선 모델 |

최종 선택 모델은 RandomForest 계열이며, 교차검증 평균 기준으로도 GridSearchCV 버전이 가장 안정적인 성능을 보였다.

## 6. GitHub 커밋 설계
이 과제는 커밋 히스토리가 단순 파일 업로드가 아니라, 개발 흐름이 드러나도록 기록하는 것이 핵심이다. 아래처럼 브랜치와 커밋을 분리하면 평가 시 의도가 명확하게 보인다.

### 6-1. 권장 브랜치 전략
- `main`: 최종 제출용 정리 상태 유지
- `feature/wine-classification`: 실제 개발 작업 브랜치
- 필요 시 `feature/report-polish`: 보고서 정리용 보조 브랜치

### 6-2. 권장 커밋 순서
| 순서 | 커밋 메시지 예시 | 커밋에 들어가는 내용 | 평가 관점에서 보이는 의미 |
| --- | --- | --- | --- |
| 1 | `Initial upload: add project scaffold and dataset plan` | 프로젝트 구조, 실행 파일 뼈대, 데이터셋 선택 메모 | 단순 업로드가 아니라 개발 시작점이 드러남 |
| 2 | `Train model: implement baseline classification pipeline` | 데이터셋 로드, train/test split, Logistic Regression 또는 Random Forest 베이스라인 | 요구사항을 충족하는 핵심 기능 구현 |
| 3 | `Evaluation: add cross-validation and confusion matrix output` | 정확도 외 교차검증, 혼동행렬, classification report 추가 | 결과를 수치적으로 해석하는 단계 |
| 4 | `Parameter tuning: add GridSearchCV for RandomForest` | 하이퍼파라미터 탐색, best params, 개선 모델 비교 | 단순 구현을 넘어 실험형 과제로 보이게 함 |
| 5 | `Report polish: finalize results and submission notes` | 보고서 정리, GitHub 주소, 실행 결과 반영 | 제출물 완성도를 높이는 마무리 단계 |

### 6-3. 커밋 메시지 작성 원칙
커밋 메시지는 단순히 `update`처럼 쓰기보다, 무엇을 왜 바꿨는지 드러나야 한다.

- 동사 + 대상 + 목적 순서로 작성
- 한 커밋에는 하나의 논리적 변경만 포함
- 평가자가 히스토리만 봐도 진행 과정을 이해할 수 있게 구성
- 기능 추가, 평가 개선, 보고서 정리를 분리

### 6-4. 실제 제출에 유리한 예시
아래처럼 기록하면 과제 제출용으로 더 자연스럽다.

- `Initial upload: add wine classification starter`
- `Train model: implement baseline and split dataset`
- `Evaluation: add CV, confusion matrix, and classification report`
- `Parameter tuning: compare tuned RandomForest performance`
- `Final polish: update report and submission notes`

## 7. 실행 결과 해석
- Logistic Regression은 표준화가 필요한 선형 모델이지만, Wine 데이터셋에서는 이미 높은 성능을 냈다.
- Random Forest는 비선형 경계에 강해 테스트셋에서 완전 정분류를 달성했다.
- GridSearchCV를 적용한 Random Forest는 교차검증 평균도 가장 높아, 성능뿐 아니라 안정성 측면에서도 가장 설득력 있는 모델이었다.

## 8. 결론
본 실습은 GitHub 브랜치와 의미 있는 commit history를 기반으로, 머신러닝 모델 개발 과정을 단계별로 정리하는 훈련이 되었다. 단순히 코드를 실행하는 수준이 아니라, 베이스라인 구성, 성능 평가, 하이퍼파라미터 탐색, 결과 정리까지 포함해 OSS 개발 흐름을 경험했다.

## 9. GitHub 저장소 주소
- 저장소 URL: 본인 GitHub 저장소 주소 입력
- 브랜치명: `feature/wine-classification` 또는 과제에 맞는 임의 브랜치명 입력