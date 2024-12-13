class Solution(object):
    def addSpaces(self, s, spaces):

        new_s = [] 
        start =  0
        for space in spaces:
            new_s.append(s[start : space])
            new_s.append(" ")
            start = space
        new_s.append(s[start:])
        return "".join(new_s)
        """
        :type s: str
        :type spaces: List[int]
        :rtype: str
        """
        
if __name__ == "__main__" :

    s = "LeetcodeHelpsMeLearn"
    spaces = [8,13,15]
    solution = Solution()
    print(solution.addSpaces(s, spaces))