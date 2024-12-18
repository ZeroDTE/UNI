class Solution(object):
    def rotateString(self, s, goal):
        if len(s) != len(goal): 
            return False 
        return goal in (s+s)[:-1]

# Move the main block outside the class and create a Solution instance
if __name__ == "__main__":
    s = "abcde"
    goal = "cdeab"
    solution = Solution()  # Create an instance
    print(solution.rotateString(s, goal))  # Call method on the instance
