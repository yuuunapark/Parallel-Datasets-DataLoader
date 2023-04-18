import random
import numpy as np

import torch
from torch._six import int_classes as _int_classes
from torch import Tensor

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized

T_co = TypeVar('T_co', covariant=True)

class Sampler(Generic[T_co]):
    def __init__(self, data_source: Optional[Sized], ratio) -> None:
        pass

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError

   


class New_Sampler(Sampler):

    data_source: Sized

    def __init__(self, data_source: Sized, ratio, random_index, N_CLASS) -> None:
        # data_source는 정렬되지 않은 데이터이다. 

        self.data_source = data_source  
        self.random_index = random_index
        self.N_CLASS = N_CLASS
        
        # ratio의 정의 :minor 데이터 개수  / major 데이터 개수   
        self.ratio = ratio 


        # major/minor class 별 샘플링 할 데이터 수 
        # major class의 개수는 항상 500 이상이다.
        self.num_data_major_class = int(1000 // (1+ self.ratio))
        self.num_data_minor_class = 1000 - self.num_data_major_class

        
        # N_DATA: 500 일반화 > N_DATA는 클래스 별 데이터개수를 뜻한다 : 전체 데이터 수 / 클래스 수
        self.N_DATA = int(len(self.data_source) / N_CLASS)




        # num_data_major_'rest'는 major class 별 샘플링 할 데이터 수에서, 원래 존재하는 class 별 데이터수를 뺀 것 
        # 으로, 500개 데이터는 기존 데이터에서 겹치지 않게 그대로 가져오고, num_data_major_rest 개 만큼은 
        # 기존데이터에서 중복없이 샘플링한다. 
        self.num_data_major_rest = self.num_data_major_class - self.N_DATA



        #________________ PEER 0의 인덱스 생성하기 

        # PEER0의 인덱스 중 '메이져' 데이터의 인덱스 생성하기 
        # 이때 list_major는 클래스 하나에 해당하는 인덱스들로 이루어진 리스트이다. (인덱스 범위: 0부터 499까지)
        # 'num_data_major_rest'개의 데이터 인덱스를 500개의 인덱스 중 중복 없이 샘플링한다. 


        list_major = []
        for _ in range(self.num_data_major_rest):
            # random.randint(a,b)는 a부터 b까지의 정수중에서 랜덤 추출하므로 b = self.N_DATA -1 이다.
            x = random.randint(0,(self.N_DATA -1))
            while x in list_major:
                x = random.randint(0,(self.N_DATA -1))

            # 중복되는 숫자는 50000을 더해준다. > 데이터를 퍼올릴때(fetch.py)에서 구분하기 위함 
            x = x + 50000 
            list_major.append(x)


        # list_major의 데이터 수는 'self.num_data_major_class'개이다. 
        list_major = list(range(0, 500)) + list_major #714개




        # minor 데이터의 인덱스 생성하기 
        # 이때 list_minor는 클래스 하나에 해당하는 인덱스리스트이다.(인덱스 범위: 0부터 499까지)
        # 'num_data_minor_class'개의 데이터 인덱스를 500개의 인덱스 중 중복 없이 샘플링한다. 
        list_minor = list(range(500- self.num_data_minor_class, 500))

     
        

        #_________major_block(메이저 인덱스 연산시 사용할 block) 만들기
        block = [0] * self.num_data_major_class
        major_block = [x + y for x, y in zip(list_major , block)] # 리스트의 원소 개수는 바뀌지 않고, 리스트의 원소끼리 연산한다.
        # print('major_block :' ,len(major_block)) #714개



        for i in range(50):
            if i == 0:
                temp_block = major_block
            else:
                add_block = [500 * (i)]* self.num_data_major_class
                # print('add_block :', len(add_block)) # 714개
                new_block = [x + y for x, y in zip(list_major , add_block)] 
                # print('new_block :', len(new_block)) # 714개

                temp_block = temp_block + new_block
                # print('temp_block', len(temp_block))

        peer0_major_indices = temp_block
        # print('peer0_major_indices', len(peer0_major_indices))
       





        #_________minor_block 만들기
        block = [25000] * self.num_data_minor_class
        minor_block = [x + y for x, y in zip(list_minor , block)] 
        # print('minor_block :' ,len(minor_block)) #286개




        # 500을 15번 더하기
        # major_block: 25000개의 원소로 이루어져있다. 
        for i in range(50):
            if i == 0:
                temp_block = minor_block
            else:
                add_block = [500 * (i)]* self.num_data_minor_class
                # print('add_block :', len(add_block)) #286개
                new_block = [x + y for x, y in zip(minor_block , add_block)] 
                # print('new_block :', len(new_block)) #286개

                temp_block = temp_block + new_block
                # print('temp_block', len(temp_block))

        peer0_minor_indices = temp_block
        # print('peer0_minor_indices', len(peer0_minor_indices))





        # 모든 major클래스에 대한 index와 모든 minor클래스에 대한 index를 하나의 리스트(total_index_list)안에 담았다. 
        total_index_list = peer0_major_indices + peer0_minor_indices

        print('total_index_list', total_index_list)
        print('total_index_list', len(total_index_list))



        # 리스트에서, 여러개의 인덱스로 데이터 불러오는 함수 정의
        def get_multiple_elements_in_list(in_list, in_indices):
            return [in_list[i] for i in in_indices]
              

        # dataloader로부터 받은 self.random_index 순서에 따라, 전체 인덱스 리스트를 shuffle하기
        # 데이터로더에서 배치단위로 데이터를 출력할 때, 한 배치에 여러 클래스의 데이터들이 랜덤으로 출력되도록 하기 위함이다.
        self.random_total_index = get_multiple_elements_in_list(total_index_list, self.random_index)




    def __iter__(self) -> Iterator[int]:     
        return iter(self.random_total_index)
    



    def __len__(self) -> int:
        return len(self.data_source)






class New_BatchSampler(Sampler[List[int]]):

    def __init__(self, New_Sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool) -> None:

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = New_Sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        
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

            # self.sampler는 new_sampler의 output을 받는다.
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
        



class New_Student_Sampler(Sampler):

    data_source: Sized

    def __init__(self, data_source, random_index) -> None:
        # data_source는 정렬되지 않은 데이터이다. 

        self.data_source = data_source  
        self.random_index = random_index



    def __iter__(self) -> Iterator[int]:     
        return iter(self.random_index)
    


    def __len__(self) -> int:
        return len(self.data_source)
        


class New_Student_BatchSampler(Sampler[List[int]]):

    def __init__(self, New_Student_Sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = New_Student_Sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:


        
        if self.drop_last:
            while True:
                try:
                    batch = [next(

                    ) for _ in range(self.batch_size)]
                    yield batch
                    

                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0

            # self.sampler는 new_sampler의 output을 받는다.
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
