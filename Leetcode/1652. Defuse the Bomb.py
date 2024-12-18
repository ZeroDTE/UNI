class Solution:
    def decrypt(self, code, k):
        # Get length of input array
        n = len(code)
        # If k is 0, return array of zeros
        if k == 0:
            return [0] * n
            
        result = []
        # If k is positive
        if k > 0:
            for i in range(n):
                total = 0
                # Sum next k elements after current position
                for j in range(k):
                    total += code[(i + j + 1) % n]  # Use modulo to handle circular array
                result.append(total)
            return result
            
        # If k is negative
        if k < 0:
            for i in range(n):
                total = 0
                # Sum previous k elements before current position
                for j in range(abs(k)):
                    total += code[(i - j - 1 + n) % n]  # Use modulo to handle circular array
                result.append(total)
            return result

# Test the solution
test = Solution()
code = [5,7,1,4]
k = 3
print(test.decrypt(code, k))
