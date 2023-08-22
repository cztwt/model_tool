import concurrent.futures
import time
import numpy as np
import pandas as pd
# number_list = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
number_list = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
df = pd.DataFrame({'cust_id': ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'e', 'e'],
                   'val': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

def evaluate_item(x, a, b, c):
        # 计算总和，这里只是为了消耗时间
        result_item = count(x)
        j = a+c
        x['val'].sum()
        # 打印输入和输出结果
        return result_item

def  count(number) :
        for i in range(0, 100000000):
                i=i+1
        return number
    
    
def process(df, a, b, c):
    # 进程池
    start_time_2 = time.time()
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
            # futures = [executor.submit(evaluate_item, item, a, b, c) for item in number_list]
            # for future in concurrent.futures.as_completed(futures):
            #         print(future.result())
            for key, vals in df.groupby('cust_id'):
                futures.append(executor.submit(evaluate_item, vals, a, b, c))
    print ("Process pool execution in " + str(time.time() - start_time_2), "seconds")

if __name__ == "__main__":
    # 顺序执行
    a, b, c = 10, 'tt', 20.0
    start_time = time.time()
    # for item in number_list:
        # print(evaluate_item(item, a, b, c))
    for key, vals in df.groupby('cust_id'):
        print(evaluate_item(vals, a, b, c))
    print("Sequential execution in " + str(time.time() - start_time), "seconds")
    # 线程池执行
    # start_time_1 = time.time()
    # with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    #         futures = [executor.submit(evaluate_item, item) for item in number_list]
    #         for future in concurrent.futures.as_completed(futures):
    #                 print(future.result())
    # print ("Thread pool execution in " + str(time.time() - start_time_1), "seconds")
    process(df, a, b, c)
