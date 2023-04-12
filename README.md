# Parallel-Datasets-DataLoader

# 기존의 데이터로더에서는 출력시, 배치단위의 데이터로더 하나를 출력한다. 
# 이 Parallel-DataLoader에서는 , peer를 4개로 나눈다. 따라서 4개의 배치단위의 데이터로더를 출력한다. 

# 각 peer가 가진 main 데이터는 다음과 같다. 
# peer0 : 0 ~ 49
# peer1 : 25 ~ 74
# peer2 : 50 ~ 99
# peer3 : 75 ~ 99, 0~ 24

# New_dataloader.py, New_sampler.py, New_fetch.py 파일을 한 폴더에 넣고 실행시키면 됩니다. 
