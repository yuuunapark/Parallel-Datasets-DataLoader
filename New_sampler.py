import random
import numpy as np
import random

import torch
from torch import Tensor

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

__all__ = [
    "BatchSampler",
    "RandomSampler",
    "Sampler",
    "SequentialSampler",
    "SubsetRandomSampler",
    "WeightedRandomSampler",
]

T_co = TypeVar('T_co', covariant=True)



class Sampler(Generic[T_co]):

    def __init__(self, data_source: Optional[Sized]) -> None:
        pass

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError





class New_Sampler(Sampler[int]):
 
    data_source: Sized

    def __init__(self, data_source: Sized, ratio) -> None:
        self.data_source = data_source  
           

        if ratio != 0:

            # ratio를 소수점 둘째자리에서 반올림한 값이 짝수가 아니라면
            if (round(ratio, 2))*100  % 2 != 0:
                self.ratio = round(ratio, 2) - 0.01

            else:
                self.ratio = ratio


        self.num_data_rest_class = int(ratio * 500)
        self.num_data_corr_class = 1000-int(ratio * 500)




        # 각 class별로 어떤 데이터를 뽑을지, 설정한 ratio에 따라, 랜덤으로 인덱스 생성하기
        list_corr = []
        for _ in range(self.num_data_corr_class):
            x = random.randint(0,500)
            list_corr.append(x)

        list_rest = []
        for _ in range(self.num_data_rest_class):
            x = random.randint(0,500)
            list_rest.append(x)

        twenty_five_hundreds = [25000]*500

        
        self.list_corr = list_corr
        self.list_rest = twenty_five_hundreds +list_rest



        # 전체 데이터에 대한 인덱스 리스트 생성
        temp_index= []

        # 다음 for문으로 만든 peer4_index는 데이터 500개중 랜덤으로 뽑은 500*p개의 인덱스 원소에,
        # 500을 한번 더하고, 500을 두번 더하고, 500을 99번 더한 후에 각 리스트들을 합친 것이다. 
        for i in range(50): # class의 개수
            five_hundreds = [500*(i)]* self.num_data_corr_class
            # temp_index : 원소 500*p개로 이루어진, 리스트
            temp_index = [x + y for x, y in zip(self.list_corr ,five_hundreds)] 

            if i ==0 :
                total_indexlist_corr = self.list_corr 
            else:
                total_indexlist_corr = total_indexlist_corr + temp_index

        for i in range(50): # class의 개수
            five_hundreds = [500*(i)]* self.num_data_rest_class
            # temp_index : 원소 500*p개로 이루어진, 리스트
            temp_index = [x + y for x, y in zip(self.list_rest,five_hundreds)] 

            if i ==0 :
                total_indexlist_rest = self.list_rest
            else:
                total_indexlist_rest = total_indexlist_rest + temp_index

        self.total_indexlist_corr = total_indexlist_corr
        self.total_indexlist_rest = total_indexlist_rest



        total_index_list = self.total_indexlist_corr + self.total_indexlist_rest
        print(total_index_list)



        # total_index_list: 인덱스들을 섞기 위해 만든 인덱스 리스트이다.
        self.total_index_list = sorted(total_index_list, key=lambda k: random.random())


        

    def __iter__(self) -> Iterator[int]:     
        return iter(self.total_index_list)


    def __len__(self) -> int:
        return len(self.data_source)






class New_BatchSampler(Sampler[List[int]]):


    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951


        
        if self.drop_last:
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield batch
                    

                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0


        
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]


    def __len__(self) -> int:
    
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]
