class Solution(object):
    def checkIfExist(self, arr):

        for i in range(len(arr)) : 
            for j in range(len(arr)) :
               
                if i != j and arr[i] == 2 * arr[j]:

                    return True
        return False



        """
        :type arr: List[int]
        :rtype: bool
        """
        
if __name__ == "__main__":
    arr = [10, 2, 5, 3]
    solution = Solution()
    print(solution.checkIfExist(arr))
