# from time import perf_counter
# from typing import NamedTuple
# import math

# NUMBERS = 5_000_000_000
# def is_prime(n: int) -> bool:
#     if n < 2:
#         return False
#     if n == 2:
#         return True
#     if n % 2 == 0:
#         return False
#     root = math.isqrt(n)
#     for i in range(3, root + 1, 2):
#         if n % i == 0:
#             return False
#     return True

# class Result(NamedTuple):
#     prime: bool
#     elapsed: float
    
# def check(n: int) -> Result:
#     t0 = perf_counter()
#     prime = is_prime(n)
#     return Result(prime, perf_counter() - t0)

# def main() -> None:
#     print(f'Checking {len(NUMBERS)} numbers sequentially:')
#     t0 = perf_counter()
#     for n in NUMBERS:
#         prime, elapsed = check(n)
#         label = 'P' if prime else ' '
#         print(f'{n:16} {label} {elapsed:9.6f}s')
#         elapsed = perf_counter() - t0
#         print(f'Total time: {elapsed:.2f}s')
    
# if __name__ == '__main__':
#     main()