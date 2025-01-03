import time

def time_to_string(input):
    time_tuple = time.gmtime(input)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time_tuple)