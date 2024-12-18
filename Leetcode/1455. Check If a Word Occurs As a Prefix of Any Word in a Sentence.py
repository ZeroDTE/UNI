class Solution(object):
    def isPrefixOfWord(self, sentence, searchWord):
        sentence_list = sentence.split()



        for i, word in  enumerate(sentence_list) : 
            if word.startswith(searchWord): 
                return (i +1)
        return -1
      
if "__name__" == "__main__": 
    sentence = "i love eating burger"
    searchWord = "burg"
    print(Solution().isPrefixOfWord(sentence, searchWord))  