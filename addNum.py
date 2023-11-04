def generate_intervals(start, end, interval_size):
    intervals = []
    current = start
    while current <= end:
        interval = [current, current + interval_size]
        intervals.append(interval)
        current += interval_size
    return intervals


def add_num(start, end):
    sum = 0
    while start <= end:
        sum += start
        start += 1
    return sum


if __name__ == '__main__':
    start = 1
    end = 10
    interval_size = 2

    intervals = generate_intervals(start, end, interval_size)
    s = 0
    for interval in intervals:
        s += add_num(interval[0], interval[1])
        print(s)
    print(intervals)

