import torch


class _BaseDatasetFetcher(object):
    def __init__(self, dataset, sorted_dataset ,N_CLASS, N_PEER,  auto_collation, collate_fn, drop_last):
        # 섞여있던 데이터셋에 인덱스를 붙인것 
        self.dataset = dataset 

        # sorted_dataset : 데이터를 class를 기준으로 오름차순 정렬하여 인덱스를 붙인 것 
        self.sorted_dataset = sorted_dataset
        self.N_CLASS = N_CLASS
        self.N_PEER = N_PEER
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last


        # N_DATA: 클래스 별 데이터 수 len(dataset)/ N_CLASS # 500개
        N_DATA= int(len(self.dataset)/ N_CLASS)
        

        # 다음은 peer0의 major_dataset, minor_dataset이다.
        self.peer0_major_dataset = self.sorted_dataset[0* N_DATA : 50 * N_DATA] 
        self.peer0_minor_dataset = self.sorted_dataset[50* N_DATA : 100 * N_DATA] 


        # peer0 데이터셋은 앞쪽에는 major데이터, 뒤쪽에는 minor데이터를 갖는다. 
        # student 데이터: 정렬하지 않은 데이터를 받는다. 
        self.peer0_dataset = self.peer0_major_dataset + self.peer0_minor_dataset       




    def fetch(self, possibly_batched_index):
        raise NotImplementedError()




class New_MapDatasetFetcher(_BaseDatasetFetcher):
    def fetch(self, index, student_index ,random_index):


        self.peer0_data = []
        self.peer1_data = []
        self.peer2_data = []
        self.stu_data = []

        # 데이터를 퍼올리는 규칙
        # sampler에서 뽑은 index로 peer0의 데이터를 뽑는다.
        # peer0에 데이터에 맞추어, peer1과 peer2 그리고 student의 데이터를 뽑는다. 



        # 여기서 index는 한 배치에 포함되어있는 인덱스 리스트이다. 
        # 예를들어 batch_size =32 인 경우, index = [10748, 834, 11046, 1941, 8230, 49362, 30960, 17828, 10845, 11003, 2521, 19009, 31794, 1549, 15982, 49377, 11877, 9808, 43542, 6182, 3874, 5340, 12747, 3135, 4137, 49359, 14797, 42328, 17806, 17387, 3425, 12710]
        for idx in index:


            #중복되는 데이터의 인덱스가 나온 경우 > (세번째)
            if idx >= 50000:
                print('세번째 경우')
                idx = idx - 50000
                self.peer0_class = self.peer0_dataset[idx][1][1]
                self.peer0_index = self.peer0_dataset[idx][0]

                # match_class는 peer0의 class와 매칭되는 클래스를 뜻한다. 
                # 만약 peer0_class = 1이면, match_class = 51이다. 

                self.match_class = self.peer0_class + 50
                self.match_index = self.peer0_index - 286


                if self.peer0_class < 16:

                    self.peer0_data.append(self.peer0_dataset[idx][1])
                    self.peer1_data.append(self.sorted_dataset[self.match_class*(500)+ self.peer0_index][1])
                    self.peer2_data.append(self.peer0_dataset[idx][1])
                    self.stu_data.append(self.sorted_dataset[self.match_class*(500)+ self.peer0_index][1])

                elif self.peer0_class < 33:

                    self.peer0_data.append(self.peer0_dataset[idx][1])
                    self.peer1_data.append(self.sorted_dataset[self.match_class*(500)+ self.peer0_index][1])
                    self.peer2_data.append(self.sorted_dataset[self.match_class*(500)+ self.peer0_index][1])
                    self.stu_data.append(self.sorted_dataset[self.match_class*(500)+ self.peer0_index][1])

                elif self.peer0_class < 50:

                    self.peer0_data.append(self.sorted_dataset[idx][1])
                    self.peer1_data.append(self.sorted_dataset[idx][1])
                    self.peer2_data.append(self.sorted_dataset[self.match_class*(500)+ self.peer0_index][1])
                    self.stu_data.append(self.sorted_dataset[self.match_class*(500)+ self.peer0_index][1])





            else:

                # peer0의 class와 index는 for문의 idx에 의해 이미 정해진 상태이다. 
                print('출력시작')
                self.student_class = self.sorted_dataset[idx][1][1]
                self.peer0_class = self.peer0_dataset[idx][1][1]
                self.student_index = self.sorted_dataset[idx][0]
                self.peer0_index = self.peer0_dataset[idx][0]


          


                # 만약 peer0의 class가 major라면 (첫번째 두번째 세번째)
                if self.peer0_class in list(range(50)):
                

                    # stu data의 인덱스가 0부터 285중 하나라면
                    if 0 <= self.student_index <= 285 :
                        print('첫번째 경우')


                        self.peer0_data.append(self.peer0_dataset[idx][1])
                        self.peer1_data.append(self.sorted_dataset[idx][1])
                        self.peer2_data.append(self.sorted_dataset[idx][1])
                        self.stu_data.append(self.sorted_dataset[idx][1])



                    # stu data의 인덱스가 286부터 499중 하나이면(두번째 경우)
                    elif self.student_index < 500: 

                        print('두번째 경우')
                        print('peer0_class', self.peer0_class)

                        # peer0의 class에 따라 peer1의 class 정하기
                        self.match_class = self.peer0_class + 50
                        print('match_class', self.match_class)

                        
                        # ratio= 0.4일 경우에 286
                        self.match_index = self.peer0_index - 286
                        # peer1_class의 peer_index 데이터 뽑아오기
                        print('match_index', self.match_index)

                        if self.peer0_class < 16:
                            print('self.peer0_class < 16:')
                            self.peer0_data.append(self.peer0_dataset[idx][1])
                            self.peer1_data.append(self.sorted_dataset[self.match_class*(500)+ self.match_index][1])
                            self.peer2_data.append(self.sorted_dataset[idx][1])
                            self.stu_data.append(self.sorted_dataset[idx][1])


                        elif self.peer0_class < 33:
                            print('self.peer0_class < 33:')

                            self.peer0_data.append(self.peer0_dataset[idx][1])
                            self.peer1_data.append(self.sorted_dataset[self.match_class*(500)+ self.match_index][1])
                            self.peer2_data.append(self.sorted_dataset[self.match_class*(500)+ self.match_index][1])
                            self.stu_data.append(self.peer0_dataset[idx][1])


                        elif self.peer0_class < 50:
                            print('self.peer0_class < 50:')

                            self.peer0_data.append(self.peer0_dataset[idx][1])
                            self.peer1_data.append(self.peer0_dataset[idx][1])
                            self.peer2_data.append(self.sorted_dataset[self.match_class*(500)+ self.match_index][1])
                            self.stu_data.append(self.peer0_dataset[idx][1])





                # peer0의 데이터가 major가 아닐때  student class와 peer0의 class가 같다면
                elif self.student_class == self.peer0_class: 
                    print('4번째 경우')
                    # peer0의 class에 따라 peer1의 class 정하기
                    self.peer1_class = self.peer0_class
                    self.peer0_data.append(self.peer0_dataset[idx][1])
                    self.peer1_data.append(self.sorted_dataset[idx][1])
                    self.peer2_data.append(self.sorted_dataset[idx][1])
                    self.stu_data.append(self.sorted_dataset[idx][1])


           



        # 피어의 개수는 self.N_PEER로 받아옴 

        if self.N_PEER == 3 :
            return self.collate_fn(self.peer0_data), self.collate_fn(self.peer1_data), self.collate_fn(self.peer2_data), self.collate_fn(self.stu_data)
        
        if self.N_PEER == 4 :
            return self.collate_fn(self.peer0_data), self.collate_fn(self.peer1_data), self.collate_fn(self.peer2_data), self.collate_fn(self.peer3_data), self.collate_fn(self.stu_data)

        if self.N_PEER == 5 :
            self.collate_fn(self.peer0_data), self.collate_fn(self.peer1_data),self.collate_fn(self.peer2_data),self.collate_fn(self.peer3_data),self.collate_fn(self.peer4_data), self.collate_fn(self.stu_data)

