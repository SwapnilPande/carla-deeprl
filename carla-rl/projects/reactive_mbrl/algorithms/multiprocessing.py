import time
from tqdm import tqdm

import torch.multiprocessing as mp


class Multiprocessor:

    def __init__(self, model, config, queue_size):
        self.queue = mp.Queue(maxsize=queue_size)
        self.model = model
        self.config = config

    def process(self, worker, works, num_processes=30):

        print('Spawning {} processors'.format(num_processes))
        processes = []

        # Spawn labeling workers
        for pid in range(num_processes):
            p = mp.Process(target=worker, args=(pid, self.queue, self.model, self.config))
            p.start()
            processes.append(p)

        print('Populating queue')

        # Fill queue as needed
        for work in tqdm(works):
            self.queue.put(work)

            while self.queue.qsize() >= num_processes:
                time.sleep(.1)

        print('Finished populating queue. Waiting for queue to empty')

        # Stall until queue is empty
        while self.queue.qsize() > 0:
            time.sleep(.1)

        # Kill processes
        for p in processes:
            p.join()

        self.queue.close()
        self.queue.join_thread()

        print('Done')
