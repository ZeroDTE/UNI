class Solution():
    def countDigits(self, num):
        num_Str = str(num)
        count = 0
        for digit in num_Str:
            if num % int(digit)== 0 : 
                count += 1
        return count
    
test = Solution()
num = 1248
print(test.countDigits(num))