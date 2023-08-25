import multiprocessing

def worker(data, queue):
    queue.put(data)

def main():
    queue = multiprocessing.Queue()
    process1 = multiprocessing.Process(target=worker, args=([1, 2, 3], queue,))
    process2 = multiprocessing.Process(target=worker, args=([4, 5, 6], queue,))
    process1.start()
    process2.start()
    process1.join()
    process2.join()

    while not queue.empty():
        print(queue.get())

if __name__ == '__main__':
    main()