def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = w1*x1+w2*x2
    if tmp > theta:
        return 1
    else:
        return 0
    

if __name__ == '__main__':
    print(AND(1,1))