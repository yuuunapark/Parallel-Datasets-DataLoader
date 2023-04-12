# 각 peer가 가진 main 데이터는 다음과 같다. 
# peer0 : 0 ~ 49
# peer1 : 25 ~ 74
# peer2 : 50 ~ 99
# peer3 : 75 ~ 99, 0~ 24



import numpy as np
import torch

from random import shuffle




class _BaseDatasetFetcher(object):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        self.dataset = dataset #정렬된 데이터를 받는다. 
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last


        # 각 클래스에 해당하는 데이터(major 클래스의 데이터)
        self.peer0_dataset_corr = self.dataset[0*500 : 50 *500]
        self.peer1_dataset_corr = self.dataset[25*500 : 75 *500]
        self.peer2_dataset_corr = self.dataset[50*500 : 100 *500]
        self.peer3_dataset_corr = self.dataset[75*500 : 100 *500] + self.dataset[0*500 :25*500]
        
        # 각 클래스의 나머지 데이터(minor 클래스의 데이터)
        self.peer0_dataset_rest = self.dataset[50*500 : 100 *500]
        self.peer1_dataset_rest = self.dataset[0*500 : 25 *500] + self.dataset[75*500 : 100 *500]
        self.peer2_dataset_rest = self.dataset[0*500 : 50 *500] 
        self.peer3_dataset_rest = self.dataset[25*500 : 75 *500] 

        
        # 피어별 데이터 모음
        self.peer0_dataset_total = self.peer0_dataset_corr + self.peer0_dataset_rest
        self.peer1_dataset_total = self.peer1_dataset_corr + self.peer1_dataset_rest
        self.peer2_dataset_total = self.peer2_dataset_corr + self.peer2_dataset_rest
        self.peer3_dataset_total = self.peer3_dataset_corr + self.peer3_dataset_rest



    def fetch(self, possibly_batched_index):
        raise NotImplementedError()



class New_MapDatasetFetcher(_BaseDatasetFetcher):
    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                data = self.dataset.__getitems__(possibly_batched_index)
            else:
                # 각 peer별 데이터셋에서 인덱스를 통해 데이터 퍼올리기
                data0 = [self.peer0_dataset_total[idx] for idx in possibly_batched_index]
                data1 = [self.peer1_dataset_total[idx] for idx in possibly_batched_index]
                data2 = [self.peer2_dataset_total[idx] for idx in possibly_batched_index]
                data3 = [self.peer3_dataset_total[idx] for idx in possibly_batched_index]

                


        else:
            data = self.dataset[possibly_batched_index]

        
        return self.collate_fn(data0), self.collate_fn(data1), self.collate_fn(data2), self.collate_fn(data3)
