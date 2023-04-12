# Parallel-Datasets-DataLoader

#### 기존의 데이터로더에서는 출력시, 배치단위의 데이터로더 하나를 출력합니다.
#### 이 Parallel-DataLoader에서는 , peer를 4개로 나눈다. 따라서 4개의 배치단위의 데이터로더를 출력합니다.
#### 이떄, 각 peer는 major class와 minor class를 가집니다.
#### New_dataloader에 'ratio' 인자를 통해 major class 데이터의 개수와 minor class 데이터의 개수를 조정할 수 있습니다. 
#### 만약 ratio = 0.9로 설정한다면 minor class에서는 각 class당 데이터를 500*0.9 = 450개 뽑고,
#### major class에서는 각 class당 데이터를 1000- 450 = 550 개 뽑습니다. 

### 각 peer가 가진 main 데이터는 다음과 같습니다. 
### peer0 : 0 ~ 49
### peer1 : 25 ~ 74
### peer2 : 50 ~ 99
### peer3 : 75 ~ 99, 0~ 24

### New_dataloader.py, New_sampler.py, New_fetch.py 파일을 한 폴더에 넣고 실행시키면 됩니다. 
