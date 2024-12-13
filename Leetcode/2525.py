class Solution(object):
    def categorizeBox(self, length, width, height, mass):
        vol = length * width * height
        cat = ""
        
        # Check if box is bulky
        if length >= 10**4 or width >= 10**4 or height >= 10**4 or vol >= 10**9:
            cat = "Bulky"
            
        # Check if box is heavy    
        if mass >= 100:
            if cat == "Bulky":
                cat = "Both"
            else:
                cat = "Heavy"
        else:
            if cat != "Bulky":
                cat = "Neither"
                
        return cat