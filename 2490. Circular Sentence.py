class Solution(object):
    def isCircularSentence(self, sentence):
        """
        :type sentence: str
        :rtype: bool
        """
        words = sentence.split()
        
        # Check if first letter of first word equals last letter of last word
        if words[0][0] != words[-1][-1]:
            return False
            
        # Check if last letter of each word equals first letter of next word
        for i in range(len(words)-1):
            if words[i][-1] != words[i+1][0]:
                return False
                
        return True