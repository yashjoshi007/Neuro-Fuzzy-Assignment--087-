import numpy as np
class Hebb :
    def __init__(self) :
        pass
    def HebbAND(self) :
        print ("\n")
        x1 = [1, 1, -1, -1]
        x2 = [1, -1, 1, -1]
        x3 = [1, 1, 1, 1]
        y = [1, -1, -1, -1] #target
        w1=0
        w2=0
        w3=0
        b=0
        w1n=0
        w2n=0
        w3n=0
        bn=0
        
        for i in range(0,4) :
            w1n = w1+x1[i]*y[i]
            w2n = w2+x2[i]*y[i]
            bn = b+y[i]
            print ("Weights and bias after iteration "+str(i)+":")
            print ("W1 :" +str(w1n))
            print ("W2 :" +str(w2n))
            print ("b  :" +str(bn))
            w1 = w1n
            w2 = w2n
            b = bn
        print ("\nFinal Weights:")
        print (w1, w2)
        print ("Final Bias:")
        print (b)
def main() :
    hebb = Hebb()
    hebb.HebbAND()
if __name__ == '__main__' :
    main()