from tabulate import tabulate # type: ignore

# Dataset generation
seed = 110
keys = [(seed*i + i*i) % 997 for i in range(1, 31)]
keys = [f"K{val}" for val in keys]

# Table size
m = 41

# Hash functions
def h1(k):
    return int(k[1:]) % m

def h2(k):
    digits = sum(int(d) for d in k[1:])
    return (digits * 3) % m

# Linear probing
def insert_linear(keys, hash_func):
    table = [None]*m
    collisions, steps = 0, 0
    for key in keys:
        idx = hash_func(key)
        if table[idx] is None:
            table[idx] = key
        else:
            collisions += 1
            while table[idx] is not None:
                steps += 1
                idx = (idx + 1) % m
            table[idx] = key
    return collisions, steps

# Separate chaining
def insert_chaining(keys, hash_func):
    table = [[] for _ in range(m)]
    collisions, steps = 0, 0
    for key in keys:
        idx = hash_func(key)
        if table[idx]:
            collisions += 1
        table[idx].append(key)
    return collisions, steps

# Run experiments
results = []
for func, name in [(h1, "h1 (k mod m)"), (h2, "h2 (digit sum × 3 mod m)")]:
    c_lin, s_lin = insert_linear(keys, func)
    c_chain, s_chain = insert_chaining(keys, func)
    results.append([name, "Linear Probing", c_lin, s_lin, round(30/m, 2)])
    results.append([name, "Separate Chaining", c_chain, s_chain, round(30/m, 2)])

# Print results as table
headers = ["Hash Function", "Method", "Collisions", "Steps", "Load Factor"]
print(tabulate(results, headers=headers, tablefmt="grid"))
