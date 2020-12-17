import multiprocessing

task = multiprocessing.JoinableQueue()
lst = [1] * 4 + [0]
print(lst)
