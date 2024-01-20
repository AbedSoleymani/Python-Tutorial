def sum(self, num):
    if num < 1:
        raise ValueError('The input should be a positive integer!')
    buffer = [0]*(num + 1)
    for index in range(1, num+1):
        buffer[index] = buffer[index-1] + index
    
    return buffer[-1]


"""
Problem:
The Climbing Stairs problem is a classic dynamic programming problem
that involves finding the number of distinct ways to climb a staircase
of n steps, taking 1 or 2 steps at a time.

Objective function:
f(num_stairs) is the function which returns total number of ways.

Base cases:
f(1) = 1 -> 1
f(2) = 2 -> 1+1, 2

Recurrence relation:
f(n) = f(n-1) + f(n-2)
"""

def climb_stairs(self, num):
    if num < 1:
        raise ValueError('The input should be a positive integer!')
    buffer = [1]*num
    buffer[1] = 2
    for i in range(2, num):
        buffer[i] = buffer[i-1] + buffer[i-2]
    return buffer[-1]



"""
Problem:
You are a professional robber planning to rob houses along a street.
Each house has a certain amount of money stashed, the only constraint
stopping you from robbing each of them is that adjacent houses have
security systems connected and it will automatically contact the police
if two adjacent houses were broken into on the same night.
Given an integer array nums representing the amount of money of each house,
return the maximum amount of money you can rob tonight without alerting the police.

Objective function:
f(nums) is the function which returns max value of stolen money.

Base cases:
f(1) = nums[0]
f(2) = max(nums[0], nums[1])

Recurrence relation:
f(n) = max(f(n-1), f(n-2) + nums[n])
"""
def rob(self, nums):
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    buffer = [0] * len(nums)
    buffer[0] = nums[0]
    buffer[1] = max(nums[0], nums[1])
    
    for i in range(2, len(nums)):
        buffer[i] = max(buffer[i-1], buffer[i-2] + nums[i])
    
    return buffer[-1]


"""
Problem:
The alternating sum of a 0-indexed array is defined as the sum of the elements
at even indices minus the sum of the elements at odd indices.
For example, the alternating sum of [4,2,5,3] is (4 + 5) - (2 + 3) = 4.
Given an array nums, return the maximum alternating sum of any subsequence of
nums (after reindexing the elements of the subsequence).
A subsequence of an array is a new array generated from the original array
by deleting some elements (possibly none) without changing the remaining elements'
relative order. For example, [2,7,4] is a subsequence of [4,2,3,7,2,1,4]
(the underlined elements), while [2,4,2] is not.

Dynamic programming logic:
For each index, we can maintain two arrays: one for the maximum alternating sum
ending at even indices and another for the maximum alternating sum ending at odd indices.
even/odd_sum[i-1] is max sub-sequence sums of vecor nums[:i] with even/odd lengths.
You will find out why we assume these two cases!
For the last element (n-1), we first calculate even_sum:
If the last element is not present in any even subsequence, the value is even_sum(n-2)
If the last element is present in all even subsequence, the value is odd_sum(n-2)+nums[n-1]. This
is because we have to create all odd subsequences from first n-1 indexes and add the last one to them!
The final even_sum[n-1] is calculated by max(even_sum[n-2], odd_sum[n-1]+nums[n-1])
Calculating odd_sum is similar:
`   max(odd_sum[n-2], even_sum[n-1]-nums[n-1])
"""
def maxAlternatingSum(self, nums):
    n = len(nums)
    even_sum = [0] * n
    odd_sum = [0] * n
    
    even_sum[0] = nums[0]
    odd_sum[0] = 0
    
    for i in range(1, n):
        even_sum[i] = max(even_sum[i - 1], odd_sum[i - 1] + nums[i])
        odd_sum[i] = max(odd_sum[i - 1], even_sum[i - 1] - nums[i])
    
    return max(even_sum[-1], odd_sum[-1])


"""
Given an integer array nums, return true if you can partition the
array into two subsets such that the sum of the elements in both
subsets is equal or false otherwise.
"""
def canPartition(self, nums):

    total_sum = sum(nums)
    
    # If the total sum is odd, it is impossible
    # to divide into two equal-sum subsets!
    if total_sum % 2 != 0:
        return False
    
    target_sum = total_sum // 2

    buffer = [False] * (target_sum + 1)
    # buffer[i] is True when there is a subset in nums with summation of i.
    buffer[0] = True # Null set's sum is zero and is a subset of nums!
    
    for num in nums:
        # j: target_sum -> target_sum-1 -> ... -> num
        for j in range(target_sum, num - 1, -1):
            buffer[j] = buffer[j] or buffer[j - num]
    
    return buffer[-1]



def findWaysToTarget(self, nums, target):
    n = len(nums)
    dp = [[0] * (2 * target + 1) for _ in range(n + 1)]
    dp[0][target] = 1

    # dp[i][j]
    # i: 1 -> 2 -> ... -> n
    for i in range(1, n + 1):
        # j: 1 -> 2 -> ... -> 2*target
        for j in range(2 * target + 1):
            if j + nums[i - 1] <= 2 * target:
                dp[i][j] += dp[i - 1][j + nums[i - 1]]
            if j - nums[i - 1] >= 0:
                dp[i][j] += dp[i - 1][j - nums[i - 1]]

    return dp[n][target]



"""
Problem:
You are given an integer array coins representing coins of different
denominations and an integer amount representing a total amount of money.
Return the fewest number of coins that you need to make up that amount.
If that amount of money cannot be made up by any combination of the coins, return -1.
You may assume that you have an infinite number of each kind of coin.
"""
def coinChange(self, coins, amount):
    # dp[i] := fewest # Of coins to make up i
    dp = [0] + [amount + 1] * amount

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)

    return -1 if dp[amount] == amount + 1 else dp[amount]



"""
Problem:
You are given an integer array coins representing coins of different
denominations and an integer amount representing a total amount of money.
Return the number of combinations that make up that amount. If that amount
of money cannot be made up by any combination of the coins, return 0.
You may assume that you have an infinite number of each kind of coin.
"""
def change(self, amount, coins):
    dp = [1] + [0] * amount

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]

    return dp[amount]
    

"""
Say that you are a traveler on a 2D grid. You begin in the top-left corner and
your goal is to travel to the bottom-right corner. You may only move down or right.
In how many ways can you travel to the goal on a grid with dimensions m * n?
Write a function gridTraveler(m, n) that calculates this.
"""
def gridTraveler(m, n, memo={}):

    if (m, n) in memo:
        return memo[(m, n)]
    
    # base cases
    if (m, n) == (1,1):
        return 1
    if m <= 0 or n <= 0:
        return 0
    
    ways_from_top = gridTraveler(m, n-1, memo)
    ways_from_left = gridTraveler(m-1, n, memo)

    memo[(m, n)] = ways_from_left + ways_from_top

    return memo[(m, n)]

"""
Write a function canSum(targetSum, numbers) that takes in a targetSum and
an array of numbers as arguments. The function should return a boolean
indicating whether or not it is possible to generate the targetSum using numbers from the array.
You may use an element of the array as many times as needed.
You may assume that all input numbers are nonnegative.
"""

def canSum(targetSum, numbers, memo={}):
    if targetSum in memo:
        return memo[targetSum]
    if targetSum == 0:
        return True
    if targetSum < 0:
        return False
    
    for number in numbers:
        if canSum(targetSum - number, numbers, memo) == True:
            memo[targetSum] = True
            return True
    
    # if non of the above work
    memo[targetSum] = False
    return False


"""
Write a function `howSum(targetSum, numbers)` that takes in a targetSum and an array
of numbers as arguments. The function should return an array containing any combination
of elements that add up to exactly the targetSum.
If there is no combination that adds up to the targetSum, then return null.
If there are multiple combinations possible, you may return any single one.
"""
def howSum(targetSum, numbers, memo={}):
    if targetSum in memo:
        return memo[targetSum]
    if targetSum == 0:
        return []
    if targetSum < 0:
        return None

    for number in numbers:
        result = howSum(targetSum - number, numbers, memo)
        if result is not None:
            memo[targetSum] = result + [number]
            return memo[targetSum]
    
    memo[targetSum] = None
    return memo[targetSum]

"""
Write a function `bestSum(targetSum, numbers)` that takes in a targetSum and
an array of numbers as arguments.
The function should return an array containing the shortest combination of numbers
that add up to exactly the targetSum.
If there is a tie for the shortest combination, you may return any one of the shortest.
"""
def bestSum(targetSum, numbers, memo={}):
    if targetSum in memo:
        return memo[targetSum]
    if targetSum == 0:
        return []
    if targetSum < 0:
        return None
    
    best_result = None
    for number in numbers:
        result = bestSum(targetSum-number, numbers, memo)
        if result is not None:
            temp_result = result + [number]
            if (best_result is None) or (len(temp_result) < len(best_result)):
                best_result = temp_result
    if best_result is not None:
        memo[targetSum] = best_result
        return memo[targetSum]
    
    # if none of the above codes work, we have no ans
    memo[targetSum] = None
    return memo[targetSum]


"""
Write a function `canConstruct(target, wordBank)` that accepts a target string and an array of strings.
The function should return a boolean indicating whether or not the target can be
constructed by concatenating elements of the 'wordBank' array.
You may reuse elements of "wordBank" as many times as needed.
"""
def canConstruct(target, wordBank, memo={}):
    if target in memo:
        return True
    if target == "":
        return True
    
    for word in wordBank:
        if (len(target) >= len(word)) and (word == target[-len(word):]):
            if canConstruct(target[:-len(word)], wordBank, memo):
                memo[target] = True
                return memo[target]
        
    # if none of the above codes returned true:
    memo[target] = False
    return memo[target]
        
    


if __name__ == "__main__":
    m, n = 3, 3
    result = gridTraveler(m, n)
    print(f"Number of ways to travel in a {m}x{n} grid: {result}")

    targetSum = 16
    numbers = [5, 3]
    result = canSum(targetSum, numbers)
    print(f"Can we sum {targetSum} using numbers {numbers}? {result}")

    targetSum = 21
    numbers = [3, 5]
    result = howSum(targetSum, numbers)
    print(f"We can sum {targetSum} using numbers {numbers} in this way: {result}")

    targetSum = 21
    numbers = [3, 5]
    result = bestSum(targetSum, numbers)
    print(f"The best sum for {targetSum} using numbers {numbers} in this way: {result}")

    print("canConstruct('abcdef', ['ab', 'abc', 'cd', 'def', 'abcd']):")
    print(canConstruct('abcdef', ['ab', 'abc', 'cd', 'def', 'abcd']))
    print("canConstruct(skateboard, [bo, rd, ate, t, ska, sk, boar]):")
    print(canConstruct('skateboard', ['bo', 'rd', 'ate', 't', 'ska', 'sk', 'boar']))