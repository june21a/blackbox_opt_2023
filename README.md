# 프로젝트 개요
## [배경] 
반도체 공정은 설비와 공정 파라미터의 조정에 따라 재현성 있는 결과를 얻을 수 있도록 파라미터화되어 있습니다. 그러나 반도체 공정이 점점 더 미세화됨에 따라 기존의 지식에 기반한 개발은 점점 어려워지고 있습니다. 이러한 복잡한 공정에서 최적의 파라미터를 찾는 일은 높은 복잡성 때문에 매우 어려운 문제입니다. 더불어, 최적화된 파라미터의 타당성을 검증하는 과정에서도 큰 비용이 소요되어 문제가 더욱 해결하기 힘듭니다. 

이를 해결하기 위해 모델 기반 최적화(model-based optimization) 방법이 사용됩니다. 모델 기반 최적화는 시뮬레이션이나 과거 데이터를 활용하여 최적의 파라미터를 찾는 방법입니다. 특히, 새로운 데이터 조회 없이 과거 데이터로부터 학습한 모델을 활용해 복잡한 공정의 입력 변수를 최적화하는 과정을 오프라인 모델 기반 최적화(Offline Model-based Optimization)라고 합니다. 

이 문제의 어려움 중 하나는 기존 데이터로만 학습해야 하므로, 데이터 분포에서 벗어난 파라미터에 대해 학습된 모델이 이를 제대로 반영하지 못할 가능성이 높다는 점입니다. 특히 최적화된 파라미터는 기존 데이터와 많이 차이가 날 가능성이 큽니다. 따라서, 데이터 분포와 최적화된 파라미터 사이의 균형점을 잘 찾는 것이 중요합니다. 

이러한 상황에서 본 대회는 반도체 공정과 같이 복잡한 Black box 문제에 대한 최적화 문제를 풀 수 있는 AI 알고리즘의 발전을 목표로 합니다.



## [주제]
Model기반 Black-box 최적화 알고리즘 개발



## [설명]
최적의 모델을 개발하기 위해, 오프라인 모델 기반 최적화 기법을 활용하여 데이터 분포와 최적화된 파라미터의 균형점을 잘 찾아야 합니다. 이를 통해 Black box 문제에 대한 AI 알고리즘의 성능을 최대한 향상시키는 것이 목표입니다.

주어진 입력 변수 x_0​부터 x_10​까지의 값을 통해 예측된 타겟 변수 y의 값 중에서 상위 10%를 찾아내고, 이 예측된 상위 10%의 데이터 중 실제 상위 5%에 해당하는 데이터가 얼마나 포함되어 있는지를 측정하고 평가합니다.


# 프로젝트 구조
```
📦project_root
┣ 📂config
┃ ┗ 📜config.yml
┣ 📂modules
┃ ┣ 📜__init__.py
┃ ┣ 📜preprocessing.py
┃ ┗ 📜utils.py
┣ 📂train_results
┣ 📜.gitignore  
┣ 📜README.md
┣ 📜catboost_grid_search.py
┣ 📜catboost_train.py
┣ 📜ensemble.py
┣ 📜inference.py
┣ 📜requirements.txt
┗ 📜submission_code.ipynb
```

# 최종 결과
- public 4등. private 10등
- 아쉽게도 수상은 실패..
![image](https://github.com/user-attachments/assets/80a8932a-69db-4b48-88a3-52c79b30cdd5)

![image](https://github.com/user-attachments/assets/b9156891-61c2-4147-a4ac-6fa16b1d959e)


# installation
```
git clone https://github.com/june21a/blackbox_opt_2023.git
cd blackbox_opt_2023
pip install -r requirements.txt
```

# 사용 방법
## train
- config/config.yml을 편집하여 feature engineering 방법을 선택한 후 다음을 실행
```
python catboost_train.py --yml_path "/path/to/your/config" --save_path "/where/to/save/model_info"
```

## inference
```
python inference.py --root_folder "/path/to/your/model_info_folder" --save_path "/where/to/save/submission.csv"
```
