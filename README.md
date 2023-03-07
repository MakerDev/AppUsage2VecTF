APPUSAGE2VECTF
===
AppUsage2Vec 모델을 tensorflow 및 tflite로 변환한 repo.

# 1. 파일구조 
    1. main.py: 
    -> AppUsage2VecTF모델을 학습 및 evaluate하는 부분. 이때 구현의 편의를 위해 pytorch의 dataloader를 사용

    2. main_tf_dataset.py:
    -> AppUsage2VecTF모델을 학습 및 evaluate하는 부분. 다만, 여기서는 tensorflow에서 제공하는 dataloader 매커니즘을 활용

    3. AppUsage2VecDataset.py:    
    -> main(_tf_dataset).py에서 활용하기 위한 데이터셋 관련 코드

    4. AppUsage2VecTFLite.py:
    -> AppUsage2Vec의 텐서플로우 구현체

    5. tflite_conversion.py:
    -> TF model을 tflite 파일로 변환하는 코드

    6. tflite_test.py:
    -> 변환된 tflite 파일의 정상작동 여부 체크를 위한 코드

    7. utils.py:
    -> 그냥 parser 코드 분리.
